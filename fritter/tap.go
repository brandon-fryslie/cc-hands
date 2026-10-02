package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"

	"fmt"
	"io"
	"log"
	"math/big"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// A tap: the child's exchanges with one server go through fritter, and a copy of each
// goes to a unix socket for whoever listens there.
//
//	fritter --tap UPSTREAM --tap-ca VARIABLE --tap-to SOCKET -- COMMAND [ARGS...]
//
// fritter is the child's HTTP proxy: it is given fritter's address in the proxy
// variables every HTTP client reads (proxied), and still names UPSTREAM as its server,
// so to the child nothing about where it talks has changed. A connection to UPSTREAM's
// host fritter answers itself, with a certificate for that host signed by an authority it
// made when it started and holds only in memory; VARIABLE names a file of the
// certificates the child is to trust besides its own, and fritter gives it one that
// holds that authority as well. Each request on that connection is forwarded to UPSTREAM
// as it came, and its reply streamed back as it comes. Every other connection goes
// through fritter unopened, to wherever it was going.
//
// Nothing about the exchange waits on the copy: a listener that is absent, slow, or gone
// costs the child nothing, and what it missed is counted and told to it with the next
// copy it does take `[LAW:nothing-unseen]`. Which variable a program reads its extra
// certificates from is the caller's knowledge, so fritter still knows nothing of Claude
// Code.
type tapping struct {
	upstream *url.URL // an origin: scheme and host, nothing more
	trust    string
	to       string
	// The certificates fritter trusts the upstream's own with; nil for the system's.
	roots *x509.CertPool
}

// The variables HTTP clients read their proxy from, each spelling, as fritter sets them.
var proxied = []string{"HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"}

// The variables HTTP clients read the hosts they reach without a proxy from, which the
// child is given empty: every connection it makes reaches fritter, and fritter reaches
// the hosts they named as directly as the child did, from its own environment.
var exempted = []string{"NO_PROXY", "no_proxy"}

// Headers that carry a credential. A copy never holds one: it is the exchange as the
// listener needs to read it, and the listener is not the one it was sent to.
var credentials = map[string]bool{
	"authorization":       true,
	"proxy-authorization": true,
	"x-api-key":           true,
	"cookie":              true,
	"set-cookie":          true,
}

// How long a copy waits on the listener - to be let in, to take each line, and at the end
// for the copies still under way once the child has gone - before it gives up on it. The
// exchange never waits on any of it; these bound what fritter holds for a listener that
// has stopped reading.
const (
	dialTimeout  = time.Second
	writeTimeout = 5 * time.Second
	drainTimeout = 2 * time.Second
)

// What a copy is: one JSON object per line, in the order the exchange happened, on a
// connection of its own. A request line, then the reply's head and each chunk of its
// bytes as they came, then its end - or, when the upstream was never reached, that.
//
// [LAW:types-are-the-program] One struct per line kind, each with the fields that kind
// has, so no line carries a field that means nothing for it.
type requested struct {
	Kind    string      `json:"kind"` // "request"
	At      float64     `json:"at"`
	Method  string      `json:"method"`
	Path    string      `json:"path"`
	Headers [][2]string `json:"headers"`
	Body    []byte      `json:"body"`
	// Copies of earlier exchanges that never reached the listener, since the last one that did.
	Lost int64 `json:"lost"`
}

type replied struct {
	Kind    string      `json:"kind"` // "response"
	At      float64     `json:"at"`
	Status  int         `json:"status"`
	Headers [][2]string `json:"headers"`
}

type chunk struct {
	Kind  string  `json:"kind"` // "bytes"
	At    float64 `json:"at"`
	Bytes []byte  `json:"bytes"`
}

// ended closes a reply: Error is empty when it was read to its end.
type ended struct {
	Kind  string  `json:"kind"` // "end"
	At    float64 `json:"at"`
	Error string  `json:"error"`
}

type unreached struct {
	Kind  string  `json:"kind"` // "unreached"
	At    float64 `json:"at"`
	Error string  `json:"error"`
}

func now() float64 {
	return float64(time.Now().UnixNano()) / 1e9
}

// parseUpstream reads UPSTREAM as the origin it names: a path on it is the child's to
// send, as it sends every other part of a request.
func parseUpstream(raw string) (*url.URL, error) {
	upstream, err := url.Parse(raw)
	if err != nil || (upstream.Scheme != "http" && upstream.Scheme != "https") || upstream.Host == "" {
		return nil, fmt.Errorf("--tap needs an http or https URL, got %q", raw)
	}
	return &url.URL{Scheme: upstream.Scheme, Host: upstream.Host}, nil
}

// authorityOf is the host and port a URL reaches, as a CONNECT names it.
func authorityOf(target *url.URL) string {
	if target.Port() != "" {
		return target.Host
	}
	port := map[string]string{"http": "80", "https": "443"}[target.Scheme]
	return net.JoinHostPort(target.Hostname(), port)
}

// Tap is a running tap: the proxy the child is given, and the count of copies lost.
type Tap struct {
	address    string
	upstream   *url.URL
	trusted    string // the file of certificates the child is given to trust
	server     *http.Server
	opened     *http.Server // serves the child's connections to the upstream, opened
	lost       atomic.Int64
	to         string
	deliveries sync.WaitGroup
}

// startTap listens on loopback and serves until closed; dir is where it keeps the file of
// certificates the child is given.
func startTap(t tapping, dir string) (*Tap, error) {
	authority, err := newAuthority(t.upstream.Hostname())
	if err != nil {
		return nil, fmt.Errorf("cannot make the certificate the child is answered with for %s: %w", t.upstream.Host, err)
	}
	trusted := filepath.Join(dir, "trusted.pem")
	if err := os.WriteFile(trusted, append(alsoTrusted(t.trust), authority.pem...), 0o600); err != nil {
		return nil, fmt.Errorf("cannot write the certificates the child is to trust: %w", err)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("cannot listen for the child's requests to %s: %w", t.upstream, err)
	}
	tap := &Tap{address: "http://" + listener.Addr().String(), upstream: t.upstream, trusted: trusted, to: t.to}
	// [LAW:no-silent-failure] Every error the proxies meet reaches the child, as a 502 or
	// a reply cut short, and the listener, in the copy. Their own log would write into the
	// child's terminal, which fritter holds in raw mode under its interface.
	quiet := log.New(io.Discard, "", 0)
	onward := &http.Transport{
		// The proxy fritter was itself given, if any: the child reaches the world as it
		// did before fritter stood in that place.
		Proxy:           http.ProxyFromEnvironment,
		TLSClientConfig: &tls.Config{RootCAs: t.roots},
		// The reply's bytes as the upstream wrote them, compressed or not: the child
		// asked for whatever encoding it asked for, and nothing is added to that.
		DisableCompression:  true,
		ForceAttemptHTTP2:   true,
		TLSHandshakeTimeout: 10 * time.Second,
		IdleConnTimeout:     90 * time.Second,
	}
	copying := &httputil.ReverseProxy{
		Rewrite: func(out *httputil.ProxyRequest) {
			out.SetURL(t.upstream)
		},
		Transport: onward,
		// Each chunk reaches the child as it arrives: a streamed reply is the point.
		FlushInterval: -1,
		ModifyResponse: func(response *http.Response) error {
			copied := copyOf(response.Request.Context())
			copied.put(replied{Kind: "response", At: now(), Status: response.StatusCode, Headers: kept(response.Header)})
			if response.StatusCode == http.StatusSwitchingProtocols {
				// The body is now a connection both ways, which the proxy needs whole to join
				// the child to it; what crosses it is another protocol, and is not copied.
				copied.put(ended{Kind: "end", At: now(), Error: "the exchange switched protocols: what followed is not copied"})
				copied.close()
				return nil
			}
			response.Body = &teed{body: response.Body, copy: copied}
			return nil
		},
		ErrorHandler: func(writer http.ResponseWriter, request *http.Request, err error) {
			copied := copyOf(request.Context())
			copied.put(unreached{Kind: "unreached", At: now(), Error: err.Error()})
			copied.close()
			http.Error(writer, fmt.Sprintf("fritter could not reach %s: %v", t.upstream, err), http.StatusBadGateway)
		},
		ErrorLog: quiet,
	}
	copied := http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		// Read whole, so the copy holds the request the upstream is sent.
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, fmt.Sprintf("fritter could not read the request: %v", err), http.StatusBadRequest)
			return
		}
		request.Body = io.NopCloser(bytes.NewReader(body))
		copied := tap.open(requested{Kind: "request", At: now(), Method: request.Method, Path: request.URL.RequestURI(), Headers: kept(request.Header), Body: body})
		copying.ServeHTTP(writer, request.WithContext(context.WithValue(request.Context(), copyKey{}, copied)))
	})
	// What the child asks of anywhere else, in plain HTTP, goes on as it was asked.
	passing := &httputil.ReverseProxy{Rewrite: func(*httputil.ProxyRequest) {}, Transport: onward, FlushInterval: -1, ErrorLog: quiet}
	opening := handing(listener.Addr())
	tap.opened = &http.Server{Handler: copied, ErrorLog: quiet}
	go tap.opened.Serve(opening)
	tapped := authorityOf(t.upstream)
	answer := &tls.Config{Certificates: []tls.Certificate{authority.leaf}, NextProtos: []string{"http/1.1"}}
	tap.server = &http.Server{
		Handler: http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			switch {
			case request.Method == http.MethodConnect && request.Host == tapped && t.upstream.Scheme == "https":
				joined(writer, func(connection net.Conn) { opening.hand(tls.Server(connection, answer)) })
			case request.Method == http.MethodConnect:
				tunnel(writer, request.Host)
			case !request.URL.IsAbs():
				http.Error(writer, "fritter is a proxy: ask it for an absolute URL, or CONNECT", http.StatusBadRequest)
			case request.URL.Scheme == t.upstream.Scheme && authorityOf(request.URL) == tapped:
				copied.ServeHTTP(writer, request)
			default:
				passing.ServeHTTP(writer, request)
			}
		}),
		ErrorLog: quiet,
	}
	go tap.server.Serve(listener)
	return tap, nil
}

// env is what the child is given: fritter as its proxy for every host, the certificates to
// trust, and what each of those variables held before, so what it runs can put that back.
func (tap *Tap) env(trust string) []string {
	given := []string{"FRITTER_TAP=" + tap.address, trust + "=" + tap.trusted}
	for _, variable := range proxied {
		given = append(given, variable+"="+tap.address)
	}
	for _, variable := range exempted {
		given = append(given, variable+"=")
	}
	for _, variable := range append(append([]string{trust}, proxied...), exempted...) {
		if outer, found := os.LookupEnv(variable); found {
			given = append(given, "FRITTER_OUTER_"+variable+"="+outer)
		}
	}
	return given
}

// alsoTrusted is what the file `variable` names holds: the certificates the child trusted
// besides its own before fritter named another file in its place.
func alsoTrusted(variable string) []byte {
	named := os.Getenv(variable)
	if named == "" {
		return nil
	}
	held, err := os.ReadFile(named)
	if err != nil {
		// The child would have been told the same, and gone on without them.
		warn("cannot read the certificates %s names, so the child is not given them: %v", variable, err)
		return nil
	}
	return append(held, '\n')
}

type authority struct {
	pem  []byte // the authority's certificate, which the child is given to trust
	leaf tls.Certificate
}

// newAuthority makes a certificate authority, and with it a certificate for host. Its
// key is never written anywhere: it lives and dies with this fritter.
func newAuthority(host string) (authority, error) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return authority{}, err
	}
	from, until := time.Now().Add(-time.Hour), time.Now().AddDate(1, 0, 0)
	root := &x509.Certificate{
		SerialNumber:          serial(),
		Subject:               pkix.Name{CommonName: fmt.Sprintf("fritter %d", os.Getpid())},
		NotBefore:             from,
		NotAfter:              until,
		IsCA:                  true,
		BasicConstraintsValid: true,
		MaxPathLenZero:        true,
		KeyUsage:              x509.KeyUsageCertSign,
	}
	signed, err := x509.CreateCertificate(rand.Reader, root, root, &key.PublicKey, key)
	if err != nil {
		return authority{}, err
	}
	root, err = x509.ParseCertificate(signed)
	if err != nil {
		return authority{}, err
	}
	leafKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return authority{}, err
	}
	leaf := &x509.Certificate{
		SerialNumber: serial(),
		Subject:      pkix.Name{CommonName: host},
		NotBefore:    from,
		NotAfter:     until,
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	if ip := net.ParseIP(host); ip != nil {
		leaf.IPAddresses = []net.IP{ip}
	} else {
		leaf.DNSNames = []string{host}
	}
	leafSigned, err := x509.CreateCertificate(rand.Reader, leaf, root, &leafKey.PublicKey, key)
	if err != nil {
		return authority{}, err
	}
	return authority{
		pem:  pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: signed}),
		leaf: tls.Certificate{Certificate: [][]byte{leafSigned}, PrivateKey: leafKey},
	}, nil
}

func serial() *big.Int {
	n, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		// crypto/rand does not fail on any system fritter runs on.
		panic(fmt.Sprintf("fritter cannot read randomness: %v", err))
	}
	return n
}

// joined answers a CONNECT and gives the connection it opened to then.
func joined(writer http.ResponseWriter, then func(net.Conn)) {
	connection, buffered, err := http.NewResponseController(writer).Hijack()
	if err != nil {
		http.Error(writer, fmt.Sprintf("fritter cannot take the connection: %v", err), http.StatusInternalServerError)
		return
	}
	if _, err := connection.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n")); err != nil {
		connection.Close()
		return
	}
	then(&read{Conn: connection, from: buffered.Reader})
}

// read is a connection whose first bytes may already have been read into a buffer.
type read struct {
	net.Conn
	from *bufio.Reader
}

func (r *read) Read(p []byte) (int, error) {
	return r.from.Read(p)
}

// tunnel joins the child to target, through the proxy fritter was given, if any.
func tunnel(writer http.ResponseWriter, target string) {
	far, err := dial(target)
	if err != nil {
		http.Error(writer, fmt.Sprintf("fritter could not reach %s: %v", target, err), http.StatusBadGateway)
		return
	}
	joined(writer, func(near net.Conn) {
		go func() {
			io.Copy(far, near)
			far.Close()
		}()
		io.Copy(near, far)
		near.Close()
	})
}

func dial(target string) (net.Conn, error) {
	via, err := http.ProxyFromEnvironment(&http.Request{URL: &url.URL{Scheme: "https", Host: target}})
	if err != nil {
		return nil, err
	}
	if via == nil {
		return net.DialTimeout("tcp", target, 10*time.Second)
	}
	if via.Scheme != "http" {
		return nil, fmt.Errorf("fritter tunnels only through an http proxy, and was given %s", via.Redacted())
	}
	connection, err := net.DialTimeout("tcp", authorityOf(via), 10*time.Second)
	if err != nil {
		return nil, err
	}
	asked := "CONNECT " + target + " HTTP/1.1\r\nHost: " + target + "\r\n"
	if via.User != nil {
		password, _ := via.User.Password()
		asked += "Proxy-Authorization: Basic " + base64.StdEncoding.EncodeToString([]byte(via.User.Username()+":"+password)) + "\r\n"
	}
	connection.SetDeadline(time.Now().Add(10 * time.Second))
	reader := bufio.NewReader(connection)
	_, err = connection.Write([]byte(asked + "\r\n"))
	var answer *http.Response
	if err == nil {
		answer, err = http.ReadResponse(reader, &http.Request{Method: http.MethodConnect})
	}
	if err == nil && answer.StatusCode != http.StatusOK {
		err = fmt.Errorf("the proxy %s answered %s", via.Redacted(), answer.Status)
	}
	if err != nil {
		connection.Close()
		return nil, err
	}
	connection.SetDeadline(time.Time{})
	return &read{Conn: connection, from: reader}, nil
}

// handed is a listener whose connections are given to it, one at a time, rather than
// accepted from the network: the child's connections to the upstream, once opened.
type handed struct {
	given  chan net.Conn
	closed chan struct{}
	once   sync.Once
	addr   net.Addr
}

func handing(addr net.Addr) *handed {
	return &handed{given: make(chan net.Conn), closed: make(chan struct{}), addr: addr}
}

func (h *handed) hand(connection net.Conn) {
	select {
	case h.given <- connection:
	case <-h.closed:
		connection.Close()
	}
}

func (h *handed) Accept() (net.Conn, error) {
	select {
	case connection := <-h.given:
		return connection, nil
	case <-h.closed:
		return nil, net.ErrClosed
	}
}

func (h *handed) Close() error {
	h.once.Do(func() { close(h.closed) })
	return nil
}

func (h *handed) Addr() net.Addr {
	return h.addr
}

// close stops serving, and lets the copies still under way finish, up to drainTimeout: the
// child has gone, and what they still hold is how its last exchanges ended.
func (tap *Tap) close() {
	tap.server.Close()
	tap.opened.Close()
	drained := make(chan struct{})
	go func() {
		tap.deliveries.Wait()
		close(drained)
	}()
	select {
	case <-drained:
	case <-time.After(drainTimeout):
	}
}

// open starts the copy of one exchange with its request, and delivers it as it grows.
func (tap *Tap) open(request requested) *exchangeCopy {
	// [LAW:nothing-unseen] What was lost is said with the next copy, and put back if that
	// one is lost too, so the count reaches the listener with the first copy it takes. A
	// copy is lost when its request never reached the listener; one that broke off after
	// was heard, and the listener sees it end broken.
	request.Lost = tap.lost.Swap(0)
	copied := newCopy()
	copied.put(request)
	tap.deliveries.Add(1)
	go func() {
		defer tap.deliveries.Done()
		if !copied.deliver(tap.to) {
			tap.lost.Add(request.Lost + 1)
		}
	}()
	return copied
}

func kept(header http.Header) [][2]string {
	pairs := [][2]string{}
	for name, values := range header {
		if credentials[strings.ToLower(name)] {
			continue
		}
		for _, value := range values {
			pairs = append(pairs, [2]string{name, value})
		}
	}
	return pairs
}

type copyKey struct{}

func copyOf(ctx context.Context) *exchangeCopy {
	return ctx.Value(copyKey{}).(*exchangeCopy)
}

// teed is a reply body whose every byte read is also put in the copy.
type teed struct {
	body   io.ReadCloser
	copy   *exchangeCopy
	failed error
	done   bool
}

func (t *teed) Read(p []byte) (int, error) {
	n, err := t.body.Read(p)
	if n > 0 {
		t.copy.put(chunk{Kind: "bytes", At: now(), Bytes: bytes.Clone(p[:n])})
	}
	switch {
	case err == io.EOF:
		t.done = true
	case err != nil:
		t.failed = err
	}
	return n, err
}

// Close ends the copy: the proxy closes the body once it has written the reply or given
// up on it, the child having hung up or the upstream having failed.
func (t *teed) Close() error {
	reason := ""
	switch {
	case t.failed != nil:
		reason = fmt.Sprintf("the reply ended early: %v", t.failed)
	case !t.done:
		reason = "the reply was not read to its end: the child hung up on it"
	}
	t.copy.put(ended{Kind: "end", At: now(), Error: reason})
	t.copy.close()
	return t.body.Close()
}

// exchangeCopy holds the lines of one exchange's copy until they are written.
//
// Unbounded, so a line is never refused and the exchange never waits: a reply the
// listener reads slowly is held here, not in the child.
type exchangeCopy struct {
	mu     sync.Mutex
	lines  [][]byte
	closed bool
	dead   bool
	wake   chan struct{}
}

func newCopy() *exchangeCopy {
	return &exchangeCopy{wake: make(chan struct{}, 1)}
}

func (c *exchangeCopy) put(line any) {
	encoded, err := json.Marshal(line)
	if err != nil {
		// Every line is a struct of strings, numbers and bytes, which always encode.
		panic(fmt.Sprintf("fritter cannot encode a copy line: %v", err))
	}
	c.mu.Lock()
	if !c.dead {
		c.lines = append(c.lines, append(encoded, '\n'))
	}
	c.mu.Unlock()
	c.signal()
}

func (c *exchangeCopy) close() {
	c.mu.Lock()
	c.closed = true
	c.mu.Unlock()
	c.signal()
}

func (c *exchangeCopy) signal() {
	select {
	case c.wake <- struct{}{}:
	default:
	}
}

// deliver writes the copy to the socket as it grows, until it is closed, and says whether
// the listener took its first line, the request. A listener that stops taking lines is
// given up on, and from then on nothing more is held for it.
func (c *exchangeCopy) deliver(to string) (heard bool) {
	connection, err := net.DialTimeout("unix", to, dialTimeout)
	if err != nil {
		c.drop()
		return false
	}
	defer connection.Close()
	for {
		<-c.wake
		c.mu.Lock()
		lines, closed := c.lines, c.closed
		c.lines = nil
		c.mu.Unlock()
		for _, line := range lines {
			connection.SetWriteDeadline(time.Now().Add(writeTimeout))
			if _, err := connection.Write(line); err != nil {
				c.drop()
				return heard
			}
			heard = true
		}
		if closed {
			return true
		}
	}
}

func (c *exchangeCopy) drop() {
	c.mu.Lock()
	c.dead, c.lines = true, nil
	c.mu.Unlock()
}
