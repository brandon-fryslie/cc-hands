package main

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"crypto/x509"
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
	return tappedTrusting(t, upstream, to, nil)
}

// tappedTrusting is tapped by a fritter that trusts the upstream's certificate with roots.
func tappedTrusting(t *testing.T, upstream string, to string, roots *x509.CertPool) *Tap {
	t.Helper()
	parsed, err := parseUpstream(upstream)
	if err != nil {
		t.Fatal(err)
	}
	tap, err := startTap(tapping{upstream: parsed, trust: "EXTRA_CA", to: to, roots: roots}, shortTempDir(t))
	if err != nil {
		t.Fatalf("startTap: %v", err)
	}
	t.Cleanup(tap.close)
	return tap
}

// child is an HTTP client as the child is one: fritter its proxy, and trusting what the
// tap gave it to trust and nothing else.
func child(t *testing.T, tap *Tap) *http.Client {
	t.Helper()
	proxy, _ := url.Parse(tap.address)
	given, err := os.ReadFile(tap.trusted)
	if err != nil {
		t.Fatalf("the child was given no certificates it can read: %v", err)
	}
	trusted := x509.NewCertPool()
	if !trusted.AppendCertsFromPEM(given) {
		t.Fatalf("the certificates the child was given hold none: %q", given)
	}
	return &http.Client{Transport: &http.Transport{Proxy: http.ProxyURL(proxy), TLSClientConfig: &tls.Config{RootCAs: trusted}}}
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

// ask is the child's request to the upstream, by way of the tap.
func ask(t *testing.T, tap *Tap, body string) (int, string) {
	t.Helper()
	request, _ := http.NewRequest("POST", tap.upstream.String()+"/v1/messages?beta=true", strings.NewReader(body))
	request.Header.Set("X-Api-Key", "secret")
	request.Header.Set("X-Session", "s1")
	response, err := child(t, tap).Do(request)
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

func TestATapIsAllItsPartsOrNone(t *testing.T) {
	for _, args := range [][]string{
		{"--tap", "https://example.com", "--tap-ca", "EXTRA_CA", "--", "true"},
		{"--tap-to", "/tmp/x.sock", "--", "true"},
		{"--tap", "https://example.com", "--tap-to", "/tmp/x.sock", "--", "true"},
		{"--tap-ca", "EXTRA_CA", "--tap-to", "/tmp/x.sock", "--", "true"},
		{"--tap", "example.com", "--tap-ca", "EXTRA_CA", "--tap-to", "/tmp/x.sock", "--", "true"},
		{"--tap", "ftp://example.com", "--tap-ca", "EXTRA_CA", "--tap-to", "/tmp/x.sock", "--", "true"},
	} {
		if _, err := parse(args); err == nil {
			t.Errorf("parse(%q) took a tap it cannot run", args)
		}
	}
	parsed, err := parse([]string{"--tap", "https://example.com/v1", "--tap-ca", "EXTRA_CA", "--tap-to", "/tmp/x.sock", "--", "true"})
	if err != nil || parsed.tap == nil || parsed.tap.upstream.String() != "https://example.com" || parsed.tap.trust != "EXTRA_CA" || parsed.tap.to != "/tmp/x.sock" {
		t.Errorf("parse of a whole tap gave %+v, %v", parsed.tap, err)
	}
}

func TestTheChildIsGivenTheTapAsItsProxyAndCanPutBackWhatItReplaced(t *testing.T) {
	dir := shortTempDir(t)
	fritter := exec.Command(os.Args[0], "-test.run=TestHelperFritter")
	fritter.Env = append(os.Environ(),
		"FRITTER_HELPER=1",
		"HTTPS_PROXY=http://outer.example:3128",
		"NO_PROXY=.example.com",
		"FRITTER_HELPER_ARGS=--socket-dir\x1f"+dir+"\x1f--tap\x1fhttps://example.com\x1f--tap-ca\x1fEXTRA_CA\x1f--tap-to\x1f"+filepath.Join(dir, "wire.sock")+"\x1f--\x1fsh\x1f-c\x1f"+
			`echo "https=$HTTPS_PROXY http=$http_proxy tap=$FRITTER_TAP outer=$FRITTER_OUTER_HTTPS_PROXY ca=$EXTRA_CA outer_ca=${FRITTER_OUTER_EXTRA_CA-unset} no=${NO_PROXY-unset}/${no_proxy-unset} outer_no=$FRITTER_OUTER_NO_PROXY"`,
	)
	terminal, err := pty.Start(fritter)
	if err != nil {
		t.Fatalf("cannot start fritter on a terminal: %v", err)
	}
	defer terminal.Close()
	printed, _ := io.ReadAll(terminal)
	fritter.Wait()
	at := `http://127\.0\.0\.1:\d+`
	if !regexp.MustCompile(`https=(` + at + `) http=(` + at + `) tap=(` + at + `) outer=http://outer\.example:3128 ca=\S+/trusted\.pem outer_ca=unset no=/ outer_no=\.example\.com\r?\n`).Match(printed) {
		t.Errorf("the child saw %q, want the tap as its proxy for every host, the certificates to trust, and what it had before", printed)
	}
}

// fritterAround runs fritter as its own process around `sh -c script`, tapping upstream,
// and returns once the process has ended.
func fritterAround(t *testing.T, dir string, upstream string, to string, script string) []byte {
	t.Helper()
	fritter := exec.Command(os.Args[0], "-test.run=TestHelperFritter")
	fritter.Env = append(os.Environ(),
		"FRITTER_HELPER=1",
		"FRITTER_HELPER_ARGS=--socket-dir\x1f"+dir+"\x1f--tap\x1f"+upstream+"\x1f--tap-ca\x1fEXTRA_CA\x1f--tap-to\x1f"+to+"\x1f--\x1fsh\x1f-c\x1f"+script,
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
	fritterAround(t, dir, slow.URL, to, `curl -s -m 0.5 --data-binary @`+body+` "`+slow.URL+`/v1/messages"`)
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
	connection.Write([]byte("GET " + upstream.URL + "/ws HTTP/1.1\r\nHost: x\r\nConnection: Upgrade\r\nUpgrade: echo\r\n\r\n"))
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

func TestAConnectionToTheUpstreamIsOpenedWithTheTapsOwnCertificateAndCopied(t *testing.T) {
	to := filepath.Join(shortTempDir(t), "wire.sock")
	copies := listening(t, to)
	upstream := httptest.NewTLSServer(streaming(t).Config.Handler)
	t.Cleanup(upstream.Close)
	roots := x509.NewCertPool()
	roots.AddCert(upstream.Certificate())
	tap := tappedTrusting(t, upstream.URL, to, roots)

	request, _ := http.NewRequest("POST", upstream.URL+"/v1/messages?beta=true", strings.NewReader(`{"q":2}`))
	request.Header.Set("X-Api-Key", "secret")
	response, err := child(t, tap).Do(request)
	if err != nil {
		t.Fatalf("the child's request over TLS through the tap failed: %v", err)
	}
	got, _ := io.ReadAll(response.Body)
	response.Body.Close()
	if response.StatusCode != 200 || !strings.Contains(string(got), `{"q":2}`) {
		t.Fatalf("the child got %d %q, want the upstream's answer", response.StatusCode, got)
	}
	if issuer := response.TLS.PeerCertificates[0].Issuer.CommonName; !strings.HasPrefix(issuer, "fritter ") {
		t.Errorf("the child was answered with a certificate from %q, want the tap's own", issuer)
	}
	lines := <-copies
	if lines[0].Path != "/v1/messages?beta=true" || string(lines[0].Body) != `{"q":2}` || lines[len(lines)-1].Kind != "end" {
		t.Errorf("the opened exchange was copied as %+v", lines)
	}
}

func TestAConnectionAnywhereElseGoesThroughUnopenedAndUncopied(t *testing.T) {
	to := filepath.Join(shortTempDir(t), "wire.sock")
	copies := listening(t, to)
	elsewhere := httptest.NewTLSServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Write([]byte("elsewhere"))
	}))
	t.Cleanup(elsewhere.Close)
	tap := tapped(t, "https://api.example.com", to)
	client := child(t, tap)
	// The child trusts the server it reached as itself, which only a connection fritter
	// never opened can show it.
	client.Transport.(*http.Transport).TLSClientConfig.RootCAs.AddCert(elsewhere.Certificate())

	response, err := client.Get(elsewhere.URL + "/")
	if err != nil {
		t.Fatalf("the child could not reach a server it does not talk to through the tap: %v", err)
	}
	got, _ := io.ReadAll(response.Body)
	response.Body.Close()
	if string(got) != "elsewhere" || !response.TLS.PeerCertificates[0].Equal(elsewhere.Certificate()) {
		t.Errorf("the child got %q from %q, want the server's own answer and certificate", got, response.TLS.PeerCertificates[0].Subject)
	}
	select {
	case lines := <-copies:
		t.Errorf("an exchange with another server was copied: %+v", lines)
	case <-time.After(300 * time.Millisecond):
	}
}

func TestTheChildStillTrustsWhatItWasGivenToTrustBefore(t *testing.T) {
	outer := filepath.Join(shortTempDir(t), "outer.pem")
	if err := os.WriteFile(outer, []byte("-----BEGIN CERTIFICATE-----\nouter\n-----END CERTIFICATE-----\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("EXTRA_CA", outer)
	tap := tapped(t, "https://api.example.com", filepath.Join(shortTempDir(t), "wire.sock"))
	given, err := os.ReadFile(tap.trusted)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(given), "-----BEGIN CERTIFICATE-----\nouter\n") || strings.Count(string(given), "BEGIN CERTIFICATE") != 2 {
		t.Errorf("the child is given %q, want what it trusted before and the tap's authority", given)
	}
}

func TestTheUpstreamsHostIsOpenedWhateverCaseItIsNamedIn(t *testing.T) {
	to := filepath.Join(shortTempDir(t), "wire.sock")
	tap := tapped(t, "https://API.Example.com", to)
	connection, err := net.Dial("tcp", strings.TrimPrefix(tap.address, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Close()
	connection.Write([]byte("CONNECT api.example.com:443 HTTP/1.1\r\nHost: api.example.com:443\r\n\r\n"))
	reader := bufio.NewReader(connection)
	if response, err := http.ReadResponse(reader, &http.Request{Method: http.MethodConnect}); err != nil || response.StatusCode != 200 {
		t.Fatalf("the CONNECT came back %v, %v", response, err)
	}
	opened := tls.Client(&read{Conn: connection, from: reader}, &tls.Config{ServerName: "api.example.com", InsecureSkipVerify: true})
	if err := opened.Handshake(); err != nil {
		t.Fatalf("the connection was not answered over TLS: %v", err)
	}
	if issuer := opened.ConnectionState().PeerCertificates[0].Issuer.CommonName; !strings.HasPrefix(issuer, "fritter ") {
		t.Errorf("api.example.com was answered by %q, want the tap's own certificate", issuer)
	}
}
