package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// A tap: the child reaches one HTTP server through fritter, and a copy of each exchange
// goes to a unix socket for whoever listens there.
//
//	fritter --tap VARIABLE=UPSTREAM --tap-to SOCKET -- COMMAND [ARGS...]
//
// The child is given VARIABLE as the address of a server fritter runs on loopback, which
// forwards each request to UPSTREAM as it came and streams the reply back as it comes.
// Nothing about the exchange waits on the copy: a listener that is absent, slow, or gone
// costs the child nothing, and what it missed is counted and told to it with the next
// copy it does take `[LAW:nothing-unseen]`. Which variable a program reads its server
// from is the caller's knowledge, so fritter still knows nothing of Claude Code.
type tapping struct {
	variable string
	upstream *url.URL
	to       string
}

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

// parseTap reads VARIABLE=UPSTREAM.
func parseTap(spec string) (variable string, upstream *url.URL, err error) {
	variable, raw, found := strings.Cut(spec, "=")
	if !found || variable == "" {
		return "", nil, fmt.Errorf("--tap takes VARIABLE=UPSTREAM, got %q", spec)
	}
	upstream, err = url.Parse(raw)
	if err != nil || (upstream.Scheme != "http" && upstream.Scheme != "https") || upstream.Host == "" {
		return "", nil, fmt.Errorf("--tap %s needs an http or https URL, got %q", variable, raw)
	}
	return variable, upstream, nil
}

// Tap is a running tap: the server the child is pointed at, and the count of copies lost.
type Tap struct {
	address    string
	server     *http.Server
	lost       atomic.Int64
	to         string
	deliveries sync.WaitGroup
}

// startTap listens on loopback and serves until closed.
func startTap(t tapping) (*Tap, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("cannot listen for the child's requests to %s: %w", t.upstream, err)
	}
	tap := &Tap{address: "http://" + listener.Addr().String(), to: t.to}
	proxy := &httputil.ReverseProxy{
		Rewrite: func(out *httputil.ProxyRequest) {
			out.SetURL(t.upstream)
		},
		Transport: &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			// The reply's bytes as the upstream wrote them, compressed or not: the child
			// asked for whatever encoding it asked for, and nothing is added to that.
			DisableCompression:  true,
			ForceAttemptHTTP2:   true,
			TLSHandshakeTimeout: 10 * time.Second,
			IdleConnTimeout:     90 * time.Second,
		},
		// Each chunk reaches the child as it arrives: a streamed reply is the point.
		FlushInterval: -1,
		ModifyResponse: func(response *http.Response) error {
			copied := copyOf(response.Request.Context())
			copied.put(replied{Kind: "response", At: now(), Status: response.StatusCode, Headers: kept(response.Header)})
			response.Body = &teed{body: response.Body, copy: copied}
			return nil
		},
		ErrorHandler: func(writer http.ResponseWriter, request *http.Request, err error) {
			copied := copyOf(request.Context())
			copied.put(unreached{Kind: "unreached", At: now(), Error: err.Error()})
			copied.close()
			http.Error(writer, fmt.Sprintf("fritter could not reach %s: %v", t.upstream, err), http.StatusBadGateway)
		},
		// [LAW:no-silent-failure] Every error the proxy meets reaches the child, as a 502
		// or a reply cut short, and the listener, in the copy. Its own log would write
		// into the child's terminal, which fritter holds in raw mode under its interface.
		ErrorLog: log.New(io.Discard, "", 0),
	}
	tap.server = &http.Server{
		Handler: http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			// Read whole, so the copy holds the request the upstream is sent.
			body, err := io.ReadAll(request.Body)
			if err != nil {
				http.Error(writer, fmt.Sprintf("fritter could not read the request: %v", err), http.StatusBadRequest)
				return
			}
			request.Body = io.NopCloser(bytes.NewReader(body))
			copied := tap.open(requested{Kind: "request", At: now(), Method: request.Method, Path: request.URL.RequestURI(), Headers: kept(request.Header), Body: body})
			proxy.ServeHTTP(writer, request.WithContext(context.WithValue(request.Context(), copyKey{}, copied)))
		}),
		ErrorLog: log.New(io.Discard, "", 0),
	}
	go tap.server.Serve(listener)
	return tap, nil
}

// close stops serving, and lets the copies still under way finish, up to drainTimeout: the
// child has gone, and what they still hold is how its last exchanges ended.
func (tap *Tap) close() {
	tap.server.Close()
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
