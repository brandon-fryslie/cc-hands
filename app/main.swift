// hands.app: the hands daemon, `hands run`, as this app's child. macOS charges a grant to the app a process was
// launched under, so the Microphone and Input Monitoring grants hands asks for are hands.app's, not a terminal's.
// The app adds no behaviour to the daemon: it lives as long as the daemon does, ends it when it is quit, and says why
// when the daemon ends in failure.
import AppKit

// [LAW:one-source-of-truth] the hands a terminal runs is the one the app runs: the user's login shell, interactive so
// it reads the same rc files a terminal does, finds `hands` on the PATH they set, and hands the daemon that PATH
// (fritter's claude shim, tmux). The shell comes from the user database, as Terminal takes it, not from the
// environment a GUI app inherits.
let shell = String(cString: getpwuid(getuid())!.pointee.pw_shell)
let command = [shell, "-l", "-i", "-c", "exec hands run"]

// ~/Library/Logs is where a Mac app's log lives, so Console.app shows it.
let log = FileManager.default.homeDirectoryForCurrentUser.appending(path: "Library/Logs/hands/hands.log")

// How many of the daemon's last lines a failure shows; the reason a start was refused is its last.
let SHOWN_LINES = 12

enum Phase {
    case running
    // Quit by the person or the system: the daemon is told to stop, and the app ends once it has.
    case quitting
}

final class Launcher: NSObject, NSApplicationDelegate {
    let daemon = Process()
    var phase = Phase.running
    // The log, open for appending; the daemon writes to it too.
    let output: FileHandle
    // Where this launch's lines begin in the log, so a failure shows this run's and no earlier one's.
    let start: UInt64

    init(output: FileHandle) throws {
        self.output = output
        start = try output.seekToEnd()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        daemon.executableURL = URL(fileURLWithPath: shell)
        daemon.arguments = Array(command.dropFirst())
        daemon.standardInput = FileHandle.nullDevice
        daemon.standardOutput = output
        daemon.standardError = output
        daemon.terminationHandler = { ended in DispatchQueue.main.async { self.ended(ended) } }
        do {
            try daemon.run()
        } catch {
            // [LAW:no-silent-failure] an app that cannot start hands says so, rather than sitting with nothing running.
            fail("hands.app could not start \(shell): \(error.localizedDescription)")
            return
        }
        // [LAW:nothing-unseen] the launch and the daemon's end, in the log beside what the daemon itself says.
        said("started \(command.joined(separator: " ")) as pid \(daemon.processIdentifier)")
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard daemon.isRunning else { return .terminateNow }
        // [LAW:single-enforcer] the daemon's own SIGTERM handling winds it down; the app waits for it to have.
        phase = .quitting
        daemon.terminate()
        return .terminateLater
    }

    func ended(_ ended: Process) {
        let how = ended.terminationReason == .exit ? "exited \(ended.terminationStatus)" : "was ended by signal \(ended.terminationStatus)"
        said("hands \(how)")
        switch phase {
        case .quitting:
            NSApp.reply(toApplicationShouldTerminate: true)
        case .running where ended.terminationReason == .exit && ended.terminationStatus == 0:
            NSApp.terminate(nil)
        case .running:
            fail("hands \(how).\n\n\(lastLines())\n\nIts full log is \(log.path).")
        }
    }

    // This launch's lines from the daemon, the newest SHOWN_LINES of them.
    func lastLines() -> String {
        do {
            let reading = try FileHandle(forReadingFrom: log)
            try reading.seek(toOffset: start)
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

func fail(_ message: String) {
    NSApp.activate(ignoringOtherApps: true)
    let alert = NSAlert()
    alert.alertStyle = .critical
    alert.messageText = "hands stopped"
    alert.informativeText = message
    alert.runModal()
    NSApp.terminate(nil)
}

// The delegate is held here: NSApplication keeps only a weak reference to it.
let launcher: Launcher
do {
    try FileManager.default.createDirectory(at: log.deletingLastPathComponent(), withIntermediateDirectories: true)
    if !FileManager.default.fileExists(atPath: log.path) {
        FileManager.default.createFile(atPath: log.path, contents: nil)
    }
    launcher = try Launcher(output: FileHandle(forWritingTo: log))
} catch {
    _ = NSApplication.shared
    fail("hands.app could not open its log, \(log.path): \(error.localizedDescription)")
    exit(1)
}
NSApplication.shared.delegate = launcher
NSApplication.shared.run()
