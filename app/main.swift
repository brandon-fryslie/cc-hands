// hands.app: the hands daemon, `hands run`, as this app's child. macOS charges a grant to the app a process was
// launched under, so the Microphone and Input Monitoring grants hands asks for are hands.app's, not a terminal's.
// The app adds no behaviour to the daemon: it lives as long as the daemon does, ends it when it is quit, and says why
// when the daemon ends in failure. It starts hands only for a person whose license key Polar, the merchant of record
// hands is sold through, says is live.
import AppKit

// Opened from Finder the app runs on the user's login shell, logs to ~/Library/Logs, and keeps the license key in
// ~/Library/Application Support; a test names its own in HANDS_APP_SHELL, HANDS_APP_LOG and HANDS_APP_LICENSE.
let given = ProcessInfo.processInfo.environment
let home = FileManager.default.homeDirectoryForCurrentUser

// [LAW:one-source-of-truth] the environment a terminal gives hands is the one the app gives it: the user's login
// shell, interactive so it reads the same rc files a terminal does, says its environment, and `hands` is found on the
// PATH they set and runs with it (fritter's claude shim, tmux). The shell comes from the user database, as Terminal
// takes it, not from the environment a GUI app inherits. The shell only says the environment and is never the
// daemon: an interactive shell ignores SIGTERM, so one standing where hands will be would not hear a quit while it
// reads its rc files.
let shell = given["HANDS_APP_SHELL"] ?? String(cString: getpwuid(getuid())!.pointee.pw_shell)
// env writes the environment to a file only it holds: the shell's own output goes to the log, and a background job its
// rc files start, which keeps the shell's stdout open, holds nothing the app waits on.
let environmentFile = FileManager.default.temporaryDirectory.appending(path: "hands.app-environment-\(getpid())")
let environmentSaid = [shell, "-l", "-i", "-c", "exec /usr/bin/env -0 > '\(environmentFile.path)'"]

// ~/Library/Logs is where a Mac app's log lives, so Console.app shows it.
let log = given["HANDS_APP_LOG"].map { URL(fileURLWithPath: $0) } ?? home.appending(path: "Library/Logs/hands/hands.log")

// The license key the person entered, and when Polar last said it was live.
let licenseFile = given["HANDS_APP_LICENSE"].map { URL(fileURLWithPath: $0) }
    ?? home.appending(path: "Library/Application Support/hands/license.json")

// How long after Polar last said a key was live the app still starts hands when it cannot reach Polar.
let GRACE: TimeInterval = 14 * 24 * 60 * 60

// What Terminal gives a shell before its rc files run, which they may change: the user's locale, and their home as the
// directory it starts in.
let terminalGives = ["LANG": "\(Locale.current.identifier.prefix { $0 != "@" }).UTF-8"]

// How many of this launch's last lines a failure shows; the reason a start was refused is its last.
let SHOWN_LINES = 12
let TAIL: UInt64 = 64_000

enum Phase {
    // Asking Polar about the license key, or the person for one: nothing to wind down, so a quit ends the app at once.
    case licensing
    // The login shell saying its environment: nothing to wind down, so a quit ends it at once.
    case reading(Process)
    case running(Process)
    // Quit by the person or the system: hands is told to stop, and the app ends once it has. Quit again, hands is
    // killed: one that does not end on SIGTERM would otherwise hold the app, and a logout, open for good.
    case quitting(Process)
    case over
}

final class Launcher: NSObject, NSApplicationDelegate {
    var phase = Phase.over
    // The log, open for appending; the shell and the daemon write to it too.
    let output: FileHandle
    // Where this launch's lines begin in the log, so a failure shows this run's and no earlier one's.
    let start: UInt64

    init(output: FileHandle) throws {
        self.output = output
        start = try output.offset()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        phase = .licensing
        let held: License?
        do {
            held = try stored()
        } catch {
            ask("The license key hands keeps in \(licenseFile.path) could not be read: \(error.localizedDescription)", key: "", held: nil)
            return
        }
        guard let held else {
            ask(nil, key: "", held: nil)
            return
        }
        check(held.key, held: held)
    }

    // Polar's answer about `key`. `held` is the license the app last saw live, whose check the grace period runs from.
    func check(_ key: String, held: License?) {
        said("license: asking Polar about the key ending \(key.suffix(6))")
        validate(key) { answer in
            switch verdict(answer, key: key, held: held, now: Date.now) {
            case .run(let license, let why):
                // [LAW:nothing-unseen] which way the check went, and on whose word: Polar's, or the grace period's.
                self.said("license: \(why)")
                do {
                    try save(license)
                } catch {
                    self.stopped("hands.app could not keep your license key in \(licenseFile.path): \(error.localizedDescription)")
                    return
                }
                self.begin()
            case .ask(let why):
                self.ask(why, key: key, held: held)
            }
        }
    }

    // The person enters the key their purchase gave them, says why the last one did not start hands, and can open
    // Polar's portal, where their key and subscription are.
    func ask(_ why: String?, key: String, held: License?) {
        said("license: asking for a key: \(why ?? "none is kept yet")")
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = "hands runs with a hands subscription"
        alert.informativeText = [why, "Enter the license key from your hands purchase. Your key and your subscription are at \(merchant.portal.absoluteString)."]
            .compactMap { $0 }.joined(separator: "\n\n")
        let field = NSTextField(string: key)
        field.placeholderString = "License key"
        field.frame = NSRect(x: 0, y: 0, width: 320, height: 24)
        alert.accessoryView = field
        alert.window.initialFirstResponder = field
        alert.addButton(withTitle: "Start hands")
        alert.addButton(withTitle: "Manage Subscription…")
        alert.addButton(withTitle: "Quit")
        switch alert.runModal() {
        case .alertFirstButtonReturn:
            check(field.stringValue.trimmingCharacters(in: .whitespacesAndNewlines), held: held)
        case .alertSecondButtonReturn:
            NSWorkspace.shared.open(merchant.portal)
            ask(why, key: field.stringValue, held: held)
        default:
            NSApp.terminate(nil)
        }
    }

    func begin() {
        let reader = Process()
        reader.executableURL = URL(fileURLWithPath: shell)
        reader.arguments = Array(environmentSaid.dropFirst())
        reader.environment = given.merging(terminalGives) { _, gives in gives }
        reader.currentDirectoryURL = home
        reader.standardInput = FileHandle.nullDevice
        reader.standardOutput = output
        reader.standardError = output
        reader.terminationHandler = { ended in onMain { self.read(ended) } }
        do {
            try reader.run()
        } catch {
            // [LAW:no-silent-failure] an app that cannot start hands says so, rather than sitting with nothing running.
            stopped("hands.app could not start your login shell, \(shell): \(error.localizedDescription)")
            return
        }
        phase = .reading(reader)
    }

    func read(_ reader: Process) {
        // The user's whole environment, its tokens among them, is not left on disk once read, or not.
        defer { forget() }
        let told: Data
        do {
            guard reader.terminationReason == .exit && reader.terminationStatus == 0 else {
                throw Ended(how: how(reader))
            }
            told = try Data(contentsOf: environmentFile)
        } catch {
            stopped("hands.app could not read the environment of your login shell, \(shell): \(error.localizedDescription).\n\n\(lastLines())\n\nIts full log is \(log.path).")
            return
        }
        // Each entry is NAME=value.
        let entries = told.split(separator: 0)
        let environment = Dictionary(entries.map { entry in
            let text = String(decoding: entry, as: UTF8.self)
            let name = text.prefix { $0 != "=" }
            return (String(name), String(text.dropFirst(name.count + 1)))
        }, uniquingKeysWith: { _, last in last })
        let daemon = Process()
        // env finds `hands` on the shell's PATH, and becomes it: the pid the app signals is hands'.
        daemon.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        daemon.arguments = ["hands", "run"]
        daemon.environment = environment
        daemon.currentDirectoryURL = home
        daemon.standardInput = FileHandle.nullDevice
        daemon.standardOutput = output
        daemon.standardError = output
        daemon.terminationHandler = { ended in onMain { self.ended(ended) } }
        do {
            try daemon.run()
        } catch {
            stopped("hands.app could not start hands: \(error.localizedDescription)")
            return
        }
        phase = .running(daemon)
        // [LAW:nothing-unseen] the launch, the PATH hands was found on, and its end, beside what the daemon itself says.
        said("started hands run as pid \(daemon.processIdentifier), from \(shell)'s PATH \(environment["PATH"] ?? "(none)")")
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        switch phase {
        case .licensing:
            return .terminateNow
        case .reading(let reader):
            kill(reader.processIdentifier, SIGKILL)
            forget()
            return .terminateNow
        case .running(let daemon):
            // [LAW:single-enforcer] the daemon's own SIGTERM handling winds it down; the app waits for it to have.
            phase = .quitting(daemon)
            daemon.terminate()
            return .terminateLater
        case .quitting(let daemon):
            said("hands has not ended on SIGTERM, and the app was quit again: killing it")
            kill(daemon.processIdentifier, SIGKILL)
            return .terminateLater
        case .over:
            return .terminateNow
        }
    }

    // A SIGTERM. A second terminate while AppKit waits on the first one's reply would end the app at once, leaving
    // hands running, so a quit already under way is the launcher's to carry on.
    func terminated() {
        switch phase {
        case .quitting:
            _ = applicationShouldTerminate(NSApp)
        case .licensing, .reading, .running, .over:
            NSApp.terminate(nil)
        }
    }

    func ended(_ ended: Process) {
        let was = phase
        phase = .over
        said("hands \(how(ended))")
        switch was {
        case .quitting:
            NSApp.reply(toApplicationShouldTerminate: true)
        case .running where ended.terminationReason == .exit && ended.terminationStatus == 0:
            NSApp.terminate(nil)
        case .running:
            stopped("hands \(how(ended)).\n\n\(lastLines())\n\nIts full log is \(log.path).")
        case .licensing, .reading, .over:
            preconditionFailure("hands ended while the app was \(was)")
        }
    }

    // [LAW:nothing-unseen] what the person is told, in the log too.
    func stopped(_ message: String) {
        phase = .over
        said("stopped: \(message)")
        fail(message)
    }

    func forget() {
        // [LAW:no-silent-failure] only a file env never wrote is expected to be missing.
        if unlink(environmentFile.path) != 0 && errno != ENOENT {
            said("could not remove \(environmentFile.path): \(String(cString: strerror(errno)))")
        }
    }

    // This launch's lines from the shell and the daemon, the newest SHOWN_LINES of them, from its last TAIL bytes.
    func lastLines() -> String {
        do {
            let reading = try FileHandle(forReadingFrom: log)
            let end = try reading.seekToEnd()
            try reading.seek(toOffset: max(start, end - min(end, TAIL)))
            let text = String(decoding: try reading.readToEnd() ?? Data(), as: UTF8.self)
            return text.split(separator: "\n").filter { !$0.hasPrefix("hands.app: ") }.suffix(SHOWN_LINES).joined(separator: "\n")
        } catch {
            return "Its log could not be read: \(error.localizedDescription)"
        }
    }

    func said(_ line: String) {
        output.write(Data("hands.app: \(Date.now.ISO8601Format()) \(line)\n".utf8))
    }
}

// [LAW:one-source-of-truth] the merchant this build sells through is its bundle's: build-app.sh writes it into
// Info.plist, and a test's bundle names a stand-in.
struct Merchant {
    // Polar's license key validation, which needs no credentials: the key and the organization it was sold by.
    let validate: URL
    let organization: String
    // The customer portal, where a person finds their key and renews their subscription.
    let portal: URL
}

// [LAW:parse-dont-validate] the bundle's merchant, parsed once as the app starts; a build that names none cannot run.
func merchantOfBundle() throws -> Merchant {
    let info = Bundle.main.infoDictionary ?? [:]
    func named(_ key: String) throws -> String {
        guard let value = info[key] as? String, !value.isEmpty else { throw Unnamed(key: key) }
        return value
    }
    func url(_ key: String) throws -> URL {
        let text = try named(key)
        guard let url = URL(string: text) else { throw Unnamed(key: key) }
        return url
    }
    return Merchant(validate: try url("HandsLicenseValidate"), organization: try named("HandsLicenseOrganization"), portal: try url("HandsLicensePortal"))
}

struct Unnamed: LocalizedError {
    let key: String
    var errorDescription: String? { "this build of hands.app names no \(key) in its Info.plist; build-app.sh writes it" }
}

// What a purchase gave the person, and when Polar last said it was live.
struct License: Codable {
    let key: String
    let validated: Date
}

// The license the app kept, or none when it has kept none yet.
func stored() throws -> License? {
    let data: Data
    do {
        data = try Data(contentsOf: licenseFile)
    } catch CocoaError.fileReadNoSuchFile {
        return nil
    }
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    return try decoder.decode(License.self, from: data)
}

func save(_ license: License) throws {
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    try FileManager.default.createDirectory(at: licenseFile.deletingLastPathComponent(), withIntermediateDirectories: true)
    // Only the person's own account reads their key: written where no one sees it half-written, readable by them alone.
    let written = licenseFile.appendingPathExtension("new")
    guard FileManager.default.createFile(atPath: written.path, contents: try encoder.encode(license), attributes: [.posixPermissions: 0o600]) else {
        throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
    guard rename(written.path, licenseFile.path) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno)!) }
}

// What Polar said of a key: live, refused with its reason, or nothing, because it could not be reached or was down.
enum Answer {
    case granted
    case refused(String)
    case unreachable(String)
}

// [LAW:effects-at-boundaries] the one request the app makes: the key and the organization, and nothing of the
// person's Claude login. Its answer comes back on the main thread.
func validate(_ key: String, then: @escaping (Answer) -> Void) {
    var request = URLRequest(url: merchant.validate, timeoutInterval: 20)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try! JSONEncoder().encode(["key": key, "organization_id": merchant.organization])
    URLSession.shared.dataTask(with: request) { data, response, error in
        let answer = error.map { .unreachable($0.localizedDescription) } ?? answered((response as! HTTPURLResponse).statusCode, data ?? Data())
        onMain { then(answer) }
    }.resume()
}

struct Validated: Decodable { let status: String }
struct Detail: Decodable { let detail: String }

func answered(_ status: Int, _ body: Data) -> Answer {
    switch status {
    case 200:
        let said = try? JSONDecoder().decode(Validated.self, from: body)
        return said?.status == "granted" ? .granted : .refused("Polar answered a key that is not granted: \(String(decoding: body, as: UTF8.self))")
    // Polar busy or down is not the person's subscription ending.
    case 429, 500...:
        return .unreachable("Polar answered HTTP \(status)")
    // Polar answers an unknown, revoked, disabled or expired key 404 and says which.
    default:
        return .refused((try? JSONDecoder().decode(Detail.self, from: body))?.detail ?? "Polar answered HTTP \(status): \(String(decoding: body, as: UTF8.self))")
    }
}

// What the app does with Polar's answer: run hands on a license, or ask the person, saying why.
enum Verdict {
    case run(License, String)
    case ask(String)
}

// A key Polar says is live runs hands. One it refuses never does. When Polar cannot be reached, the key the app last
// saw live runs hands until the grace period since that check runs out; a key never seen live does not.
func verdict(_ answer: Answer, key: String, held: License?, now: Date) -> Verdict {
    switch answer {
    case .granted:
        return .run(License(key: key, validated: now), "Polar says the key ending \(key.suffix(6)) is live")
    case .refused(let why):
        return .ask("Polar did not accept the license key ending \(key.suffix(6)); if your subscription ended, renew it and start hands again. Polar says: \(why)")
    case .unreachable(let why):
        guard let held, held.key == key else {
            return .ask("hands could not reach Polar to check your license key: \(why). Connect to the internet and try again.")
        }
        let left = (held.validated + GRACE).timeIntervalSince(now)
        guard left > 0 else {
            return .ask("hands has not been able to check your subscription with Polar since \(held.validated.formatted(date: .long, time: .shortened)), and runs offline for \(Int(GRACE / 86_400)) days: \(why). Connect to the internet and try again.")
        }
        return .run(held, "could not reach Polar (\(why)); running on its word of \(held.validated.ISO8601Format()), \(Int(left / 86_400)) whole days of grace left")
    }
}

// Work for the main thread, done by its run loop in any mode: a quit waiting on hands and an alert each spin it in a
// mode of their own, and the main dispatch queue not at all while either was entered from it.
func onMain(_ work: @escaping () -> Void) {
    CFRunLoopPerformBlock(CFRunLoopGetMain(), CFRunLoopMode.commonModes.rawValue, work)
    CFRunLoopWakeUp(CFRunLoopGetMain())
}

// A process that ended other than with exit 0, as an error that says how.
struct Ended: LocalizedError {
    let how: String
    var errorDescription: String? { "it \(how)" }
}

func how(_ ended: Process) -> String {
    ended.terminationReason == .exit ? "exited \(ended.terminationStatus)" : "was ended by signal \(ended.terminationStatus)"
}

func fail(_ message: String) {
    NSApp.activate(ignoringOtherApps: true)
    let alert = NSAlert()
    alert.alertStyle = .critical
    alert.messageText = "hands stopped"
    alert.informativeText = message
    alert.runModal()
    NSApp.terminate(nil)
}

// Past this many bytes a launch begins the log again, keeping the last one beside it, so launches do not grow it
// without end.
let LOG_LIMIT: UInt64 = 10_000_000

func appending(to url: URL) throws -> FileHandle {
    // O_APPEND: every write lands at the log's end, whoever else writes to it.
    let descriptor = open(url.path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0o644)
    guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno)!) }
    return FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
}

// The delegate is held here: NSApplication keeps only a weak reference to it.
let launcher: Launcher
let merchant: Merchant
do {
    merchant = try merchantOfBundle()
} catch {
    _ = NSApplication.shared
    fail("hands.app cannot start hands: \(error.localizedDescription)")
    exit(1)
}
do {
    try FileManager.default.createDirectory(at: log.deletingLastPathComponent(), withIntermediateDirectories: true)
    var output = try appending(to: log)
    let begun = try output.seekToEnd() > LOG_LIMIT
    if begun {
        guard rename(log.path, log.path + ".1") == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno)!) }
        output = try appending(to: log)
    }
    launcher = try Launcher(output: output)
    if begun {
        launcher.said("began this log again; the last one is \(log.path).1")
    }
} catch {
    _ = NSApplication.shared
    fail("hands.app could not open its log, \(log.path): \(error.localizedDescription)")
    exit(1)
}
NSApplication.shared.delegate = launcher

// [LAW:single-enforcer] a SIGTERM (killall, pkill) quits the app as its Quit does, so hands is wound down, not orphaned.
// The handler that does nothing keeps the signal from ending the app before the source hears it. It is not SIG_IGN,
// which every child would inherit: hands would ignore the SIGTERM a quit sends it. A caught signal is reset by exec.
signal(SIGTERM) { _ in }
let terminated = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global())
terminated.setEventHandler { onMain { launcher.terminated() } }
terminated.resume()

NSApplication.shared.run()
