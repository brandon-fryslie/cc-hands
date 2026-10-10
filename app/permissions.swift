// The macOS permissions hands needs, what macOS says of each one for hands.app, and how hands asks for it.
//
// The app asks these questions in a new process of its own each time, `hands --permissions`: macOS charges the answer to
// hands.app, which is responsible for that process, and a new process never reuses an earlier answer. A process that
// is already running can keep the answer it had when Input Monitoring was turned on (measured in src/hands/voice/grant.py).
import AVFoundation
import IOKit.hid

// What macOS says of one permission for hands.app. A request shows macOS's dialog only while the permission is
// undecided. Once the person has answered, macOS shows nothing until the permission is reset. Restricted: the Mac's
// administrator does not allow it, and no answer of the person's changes that.
enum Access: String {
    case undecided, denied, granted, restricted
}

struct Permission {
    // tccutil's name for the permission, also the name `hands --permissions` uses for it.
    let service: String
    // The name System Settings lists it under.
    let name: String
    // What the permission lets hands do, in one sentence.
    let purpose: String
    // How to answer macOS's request.
    let answer: String
    let access: () -> Access
    // Shows macOS's request, and returns once the request no longer needs this process.
    let request: () -> Void
}

// In the order the setup asks for them. hands types into sessions through fritter, which needs no Accessibility grant.
let PERMISSIONS = [
    Permission(
        service: "Microphone",
        name: "Microphone",
        purpose: "hands hears what you say while you hold the talk key.",
        answer: "When macOS asks, choose Allow.",
        access: {
            switch AVCaptureDevice.authorizationStatus(for: .audio) {
            case .notDetermined: .undecided
            case .authorized: .granted
            case .denied: .denied
            case .restricted: .restricted
            @unknown default: .denied
            }
        },
        request: {
            // macOS's dialog belongs to the process that asked, so this process waits for the person's answer.
            let answered = DispatchSemaphore(value: 0)
            AVCaptureDevice.requestAccess(for: .audio) { _ in answered.signal() }
            answered.wait()
        }
    ),
    Permission(
        service: "ListenEvent",
        name: "Input Monitoring",
        purpose: "hands notices when you hold the talk key, Right Shift, in any app.",
        answer: "When macOS asks, choose Open System Settings, then turn on hands.",
        access: {
            let access = IOHIDCheckAccess(kIOHIDRequestTypeListenEvent)
            return access == kIOHIDAccessTypeGranted ? .granted : access == kIOHIDAccessTypeDenied ? .denied : .undecided
        },
        // The dialog offers to open the Input Monitoring pane, where hands is already listed. It stays up after this
        // process ends.
        request: { _ = IOHIDRequestAccess(kIOHIDRequestTypeListenEvent) }
    ),
]

// What `hands --permissions` was asked to do.
enum Asked {
    // One line per permission: its service and its access.
    case accesses
    // Reset the permission if the person denied it, so that macOS shows its request again, then show the request.
    case request(Permission)
}

struct Unasked: LocalizedError {
    let arguments: [String]
    var errorDescription: String? {
        "usage: hands --permissions [request \(PERMISSIONS.map(\.service).joined(separator: "|"))] (given: \(arguments.joined(separator: " ")))"
    }
}

// [LAW:parse-dont-validate] the arguments after `--permissions`, as what they ask.
func asked(_ arguments: [String]) throws -> Asked {
    if arguments.isEmpty {
        return .accesses
    }
    guard arguments.count == 2, arguments[0] == "request", let permission = PERMISSIONS.first(where: { $0.service == arguments[1] }) else {
        throw Unasked(arguments: arguments)
    }
    return .request(permission)
}

func permissionsCommand(_ arguments: [String]) -> Int32 {
    do {
        switch try asked(arguments) {
        case .accesses:
            for permission in PERMISSIONS {
                print(permission.service, permission.access().rawValue)
            }
        case .request(let permission):
            // [LAW:single-enforcer] a request ends with the app that asked for it, however the app ended: a Microphone
            // request otherwise waits on its dialog after a quit.
            let app = getppid()
            let parent = DispatchSource.makeProcessSource(identifier: app, eventMask: .exit, queue: .global())
            parent.setEventHandler { exit(1) }
            parent.resume()
            // An app that ended before the watch began is not reported by it: this process has a new parent by then.
            guard getppid() == app else { return 1 }
            switch permission.access() {
            case .undecided, .granted, .restricted:
                break
            case .denied:
                // tccutil finds hands.app through Spotlight, so it fails for a bundle Spotlight does not index.
                let reset = Process()
                reset.executableURL = URL(fileURLWithPath: "/usr/bin/tccutil")
                reset.arguments = ["reset", permission.service, Bundle.main.bundleIdentifier!]
                reset.standardOutput = FileHandle.nullDevice
                try reset.run()
                reset.waitUntilExit()
                // [LAW:no-silent-failure] a request after a failed reset would show nothing; tccutil has said why on stderr.
                guard reset.terminationStatus == 0 else { return 1 }
            }
            permission.request()
        }
        return 0
    } catch {
        FileHandle.standardError.write(Data("hands --permissions: \(error.localizedDescription)\n".utf8))
        return 1
    }
}
