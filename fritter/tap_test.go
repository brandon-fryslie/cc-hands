package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/creack/pty"
)

// line is any copy line, read back with every field any kind has.
type line struct {
	Kind    string      `json:"kind"`
	Method  string      `json:"method"`
	Path    string      `json:"path"`
	Headers [][2]string `json:"headers"`
	Body    []byte      `json:"body"`
	Lost    int64       `json:"lost"`
	Status  int         `json:"status"`
	Bytes   []byte      `json:"bytes"`
	Error   string      `json:"error"`
}

// listening is a socket that takes copies, each connection's lines handed over as it closes.
func listening(t *testing.T, path string) <-chan []line {
	t.Helper()
	return listeningAfter(t, path, 0)
}

// listeningAfter is listening by a listener that lets each connection wait `delay` before
// it starts reading.
func listeningAfter(t *testing.T, path string, delay time.Duration) <-chan []line {
	t.Helper()
	listener, err := net.Listen("unix", path)
	if err != nil {
		t.Fatalf("cannot listen on %s: %v", path, err)
	}
	t.Cleanup(func() { listener.Close() })
	copies := make(chan []line, 16)
	go func() {
		for {
			connection, err := listener.Accept()
			if err != nil {
				return
			}
			go func() {
				defer connection.Close()
				time.Sleep(delay)
				var lines []line
				scanner := bufio.NewScanner(connection)
				scanner.Buffer(make([]byte, 1<<20), 1<<24)
				for scanner.Scan() {
					var read line
					if err := json.Unmarshal(scanner.Bytes(), &read); err != nil {
						t.Errorf("a copy line is not JSON: %.200q", scanner.Text())
					}
					lines = append(lines, read)
				}
				copies <- lines
			}()
		}
	}()
	return copies
}

func tapped(t *testing.T, upstream string, to string) *Tap {
	t.Helper()
	parsed, err := url.Parse(upstream)
	if err != nil {
		t.Fatal(err)
	}
	tap, err := startTap(tapping{variable: "BASE_URL", upstream: parsed, to: to})
	if err != nil {
		t.Fatalf("startTap: %v", err)
	}
	t.Cleanup(tap.close)
	return tap
}

func streaming(t *testing.T) *httptest.Server {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, _ := io.ReadAll(request.Body)
		if request.Header.Get("X-Api-Key") != "secret" {
			http.Error(writer, "the credential did not reach the upstream", http.StatusUnauthorized)
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		writer.WriteHeader(http.StatusOK)
		for _, part := range []string{"event: a\ndata: " + string(body) + "\n\n", "event: b\ndata: {}\n\n"} {
			writer.Write([]byte(part))
			writer.(http.Flusher).Flush()
		}
	}))
	t.Cleanup(upstream.Close)
	return upstream
}

func ask(t *testing.T, tap *Tap, body string) (int, string) {
	t.Helper()
	request, _ := http.NewRequest("POST", tap.address+"/v1/messages?beta=true", strings.NewReader(body))
	request.Header.Set("X-Api-Key", "secret")
	request.Header.Set("X-Session", "s1")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatalf("the child's request through the tap failed: %v", err)
	}
	defer response.Body.Close()
	got, _ := io.ReadAll(response.Body)
	return response.StatusCode, string(got)
}

func header(pairs [][2]string, name string) (string, bool) {
	for _, pair := range pairs {
		if strings.EqualFold(pair[0], name) {
			return pair[1], true
		}
	}
	return "", false
}

func TestTheChildIsAnsweredByTheUpstreamAndTheExchangeIsCopiedInOrder(t *testing.T) {
	to := filepath.Join(shortTempDir(t), "wire.sock")
	copies := listening(t, to)
	tap := tapped(t, streaming(t).URL, to)

	status, got := ask(t, tap, `{"q":1}`)
	if want := "event: a\ndata: {\"q\":1}\n\nevent: b\ndata: {}\n\n"; status != 200 || got != want {
		t.Fatalf("the child got %d %q, want the upstream's 200 %q", status, got, want)
	}

	lines := <-copies
	kinds := []string{}
	var streamed string
	for _, read := range lines {
		kinds = append(kinds, read.Kind)
		streamed += string(read.Bytes)
	}
	if kinds[0] != "request" || kinds[1] != "response" || kinds[len(kinds)-1] != "end" {
		t.Fatalf("the copy's lines are %v, want request, response, bytes..., end", kinds)
	}
	request, response, end := lines[0], lines[1], lines[len(lines)-1]
	if request.Method != "POST" || request.Path != "/v1/messages?beta=true" || string(request.Body) != `{"q":1}` {
		t.Errorf("the copied request is %s %s %q", request.Method, request.Path, request.Body)
	}
	if session, _ := header(request.Headers, "X-Session"); session != "s1" {
		t.Errorf("the copied request lost its headers: %v", request.Headers)
	}
	// The listener is not who the credential was for.
	if _, found := header(request.Headers, "X-Api-Key"); found {
		t.Errorf("the copy carries the child's credential: %v", request.Headers)
	}
	if kind, _ := header(response.Headers, "Content-Type"); response.Status != 200 || kind != "text/event-stream" {
		t.Errorf("the copied reply head is %d %v", response.Status, response.Headers)
	}
	if streamed != got {
		t.Errorf("the copy's bytes are %q, the child's %q", streamed, got)
	}
	if end.Error != "" {
		t.Errorf("a reply read to its end was copied as ending %q", end.Error)
	}
}

func TestNoListenerCostsTheChildNothingAndIsToldWithTheNextCopy(t *testing.T) {
	to := filepath.Join(shortTempDir(t), "wire.sock")
	tap := tapped(t, streaming(t).URL, to)

	for range 2 {
		if status, _ := ask(t, tap, `{}`); status != 200 {
			t.Fatalf("with nobody listening the child got %d", status)
		}
	}
	// The lost copies are counted once their delivery gives up, off the child's path.
	deadline := time.Now().Add(5 * time.Second)
	for tap.lost.Load() != 2 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}

	copies := listening(t, to)
	ask(t, tap, `{}`)
	if lost := (<-copies)[0].Lost; lost != 2 {
		t.Errorf("the first copy taken says %d were lost, want 2", lost)
	}
	ask(t, tap, `{}`)
	if lost := (<-copies)[0].Lost; lost != 0 {
		t.Errorf("a copy after one taken says %d were lost, want 0", lost)
	}
}

func TestAnUpstreamThatCannotBeReachedIsA502AndCopiedAsUnreached(t *testing.T) {
	to := filepath.Join(shortTempDir(t), "wire.sock")
	copies := listening(t, to)
	closed := httptest.NewServer(http.NotFoundHandler())
	address := closed.URL
	closed.Close()
	tap := tapped(t, address, to)

	if status, _ := ask(t, tap, `{}`); status != http.StatusBadGateway {
		t.Errorf("an unreachable upstream reached the child as %d, want 502", status)
	}
	lines := <-copies
	if last := lines[len(lines)-1]; last.Kind != "unreached" || last.Error == "" {
		t.Errorf("the copy ends %+v, want unreached with its error", last)
	}
}

func TestATapIsBothHalvesOrNone(t *testing.T) {
	for _, args := range [][]string{
		{"--tap", "BASE_URL=https://example.com", "--", "true"},
		{"--tap-to", "/tmp/x.sock", "--", "true"},
		{"--tap", "BASE_URL", "--tap-to", "/tmp/x.sock", "--", "true"},
		{"--tap", "BASE_URL=ftp://example.com", "--tap-to", "/tmp/x.sock", "--", "true"},
	} {
		if _, err := parse(args); err == nil {
			t.Errorf("parse(%q) took a tap it cannot run", args)
		}
	}
	parsed, err := parse([]string{"--tap", "BASE_URL=https://example.com/", "--tap-to", "/tmp/x.sock", "--", "true"})
	if err != nil || parsed.tap == nil || parsed.tap.variable != "BASE_URL" || parsed.tap.upstream.Host != "example.com" || parsed.tap.to != "/tmp/x.sock" {
		t.Errorf("parse of a whole tap gave %+v, %v", parsed.tap, err)
	}
}

func TestTheChildIsGivenTheTapsAddressInItsVariable(t *testing.T) {
	dir := shortTempDir(t)
	fritter := exec.Command(os.Args[0], "-test.run=TestHelperFritter")
	fritter.Env = append(os.Environ(),
		"FRITTER_HELPER=1",
		"FRITTER_HELPER_ARGS=--socket-dir\x1f"+dir+"\x1f--tap\x1fBASE_URL=https://example.com\x1f--tap-to\x1f"+filepath.Join(dir, "wire.sock")+"\x1f--\x1fsh\x1f-c\x1fecho \"at=$BASE_URL\"",
	)
	terminal, err := pty.Start(fritter)
	if err != nil {
		t.Fatalf("cannot start fritter on a terminal: %v", err)
	}
	defer terminal.Close()
	printed, _ := io.ReadAll(terminal)
	fritter.Wait()
	if !regexp.MustCompile(`at=http://127\.0\.0\.1:\d+\r?\n`).Match(printed) {
		t.Errorf("the child saw %q, want BASE_URL at the tap on loopback", printed)
	}
}

// fritterAround runs fritter as its own process around `sh -c script`, tapping BASE_URL
// toward upstream, and returns once the process has ended.
func fritterAround(t *testing.T, dir string, upstream string, to string, script string) []byte {
	t.Helper()
	fritter := exec.Command(os.Args[0], "-test.run=TestHelperFritter")
	fritter.Env = append(os.Environ(),
		"FRITTER_HELPER=1",
		"FRITTER_HELPER_ARGS=--socket-dir\x1f"+dir+"\x1f--tap\x1fBASE_URL="+upstream+"\x1f--tap-to\x1f"+to+"\x1f--\x1fsh\x1f-c\x1f"+script,
	)
	terminal, err := pty.Start(fritter)
	if err != nil {
		t.Fatalf("cannot start fritter on a terminal: %v", err)
	}
	defer terminal.Close()
	printed, _ := io.ReadAll(terminal)
	fritter.Wait()
	return printed
}

func TestTheChildsLastExchangeIsCopiedToItsEndThoughFritterEndsWithTheChild(t *testing.T) {
	// The process has to end for this to show: the child hangs up mid-reply and exits, and
	// fritter, exiting with it, still has the end of that exchange's copy to write.
	if _, err := exec.LookPath("curl"); err != nil {
		t.Skip("needs curl to be the child")
	}
	dir := shortTempDir(t)
	to := filepath.Join(dir, "wire.sock")
	// A request far larger than the socket's buffer, to a listener slow to read it: the
	// copy is still being written when the child has gone.
	copies := listeningAfter(t, to, time.Second)
	body := filepath.Join(dir, "body")
	if err := os.WriteFile(body, bytes.Repeat([]byte("x"), 1<<20), 0o600); err != nil {
		t.Fatal(err)
	}
	slow := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		io.Copy(io.Discard, request.Body)
		writer.WriteHeader(http.StatusOK)
		writer.Write([]byte("first"))
		writer.(http.Flusher).Flush()
		<-request.Context().Done()
	}))
	t.Cleanup(slow.Close)
	fritterAround(t, dir, slow.URL, to, `curl -s -m 0.5 --data-binary @`+body+` "$BASE_URL/v1/messages"`)
	select {
	case lines := <-copies:
		if last := lines[len(lines)-1]; last.Kind != "end" || last.Error == "" {
			t.Errorf("the last exchange's copy ends %+v, want its end, cut short by the child", last)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no copy of the child's last exchange reached the listener")
	}
}

func TestACopyHeardAndThenBrokenOffIsNotLost(t *testing.T) {
	// The listener heard the request, and with it the count of copies lost before it; it
	// sees this copy end broken. Counting it lost as well would tell the loss twice.
	to := filepath.Join(shortTempDir(t), "wire.sock")
	listener, err := net.Listen("unix", to)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	firsts := make(chan string, 4)
	go func() {
		for {
			connection, err := listener.Accept()
			if err != nil {
				return
			}
			first, _ := bufio.NewReader(connection).ReadString('\n')
			firsts <- first
			connection.Close()
		}
	}()
	tap := tapped(t, streaming(t).URL, to)
	tap.lost.Store(3)

	ask(t, tap, `{}`)
	<-firsts
	tap.deliveries.Wait()
	if lost := tap.lost.Load(); lost != 0 {
		t.Errorf("after a copy whose request was heard, %d are counted lost, want 0", lost)
	}
}

func TestAnUpgradedExchangeJoinsTheChildToTheUpstreamAndIsCopiedToItsHead(t *testing.T) {
	to := filepath.Join(shortTempDir(t), "wire.sock")
	copies := listening(t, to)
	// An upstream that switches protocols and then echoes what it is sent.
	upstream := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		connection, buffered, err := writer.(http.Hijacker).Hijack()
		if err != nil {
			t.Errorf("cannot hijack: %v", err)
			return
		}
		defer connection.Close()
		buffered.WriteString("HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: echo\r\n\r\n")
		buffered.Flush()
		io.Copy(connection, buffered)
	}))
	t.Cleanup(upstream.Close)
	tap := tapped(t, upstream.URL, to)

	connection, err := net.Dial("tcp", strings.TrimPrefix(tap.address, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Close()
	connection.Write([]byte("GET /ws HTTP/1.1\r\nHost: x\r\nConnection: Upgrade\r\nUpgrade: echo\r\n\r\n"))
	reader := bufio.NewReader(connection)
	response, err := http.ReadResponse(reader, nil)
	if err != nil || response.StatusCode != http.StatusSwitchingProtocols {
		t.Fatalf("the child's upgrade came back %v, %v; want 101", response, err)
	}
	connection.Write([]byte("ping\n"))
	if echoed, _ := reader.ReadString('\n'); echoed != "ping\n" {
		t.Errorf("across the upgraded connection the child got %q, want its ping back", echoed)
	}
	connection.Close()

	lines := <-copies
	kinds := []string{}
	for _, read := range lines {
		kinds = append(kinds, read.Kind)
	}
	if strings.Join(kinds, ",") != "request,response,end" || lines[1].Status != 101 || lines[2].Error == "" {
		t.Errorf("the upgraded exchange was copied as %v", lines)
	}
}
