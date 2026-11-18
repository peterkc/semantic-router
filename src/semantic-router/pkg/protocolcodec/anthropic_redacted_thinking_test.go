package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const redactedBlocks = `[{"type":"thinking","thinking":"thought","signature":"signed"},{"type":"redacted_thinking","data":"opaque+/="},{"type":"text","text":"answer"}]`
const redactedRequest = `{"model":"m","max_tokens":16,"messages":[{"role":"assistant","content":` + redactedBlocks + `},{"role":"user","content":"next"}]}`
const redactedResponse = `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":` + redactedBlocks + `,"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":3}}`
const messagesFormat = llmprotocol.AnthropicMessagesV1

func TestAnthropicRedactedRequest(t *testing.T) {
	result, err := NewBuiltinEngine().TranslateRequest(messagesFormat, messagesFormat, []byte(redactedRequest), func(r *llmprotocol.Request) error { r.Model = "backend"; return nil })
	require.NoError(t, err)
	var wire anthropicRequestWire
	require.NoError(t, json.Unmarshal(result.Body, &wire))
	require.JSONEq(t, redactedBlocks, string(wire.Messages[0].Content))
}

func TestAnthropicRedactedResponse(t *testing.T) {
	result, err := NewBuiltinEngine().TranslateResponse(messagesFormat, messagesFormat, []byte(redactedResponse), func(r *llmprotocol.Response) error { r.Model = "public"; return nil })
	require.NoError(t, err)
	var wire anthropicResponseWire
	require.NoError(t, json.Unmarshal(result.Body, &wire))
	require.JSONEq(t, redactedBlocks, string(wire.Content))
}

const redactedStream = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"thought"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"opaque+/="}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"answer"}}

event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}

event: message_stop
data: {"type":"message_stop"}

`

func TestAnthropicRedactedStream(t *testing.T) {
	engine := NewBuiltinEngine()
	ctx := llmprotocol.StreamContext{Context: context.Background(), PublicModel: "m"}
	stream, err := engine.NewStream(messagesFormat, messagesFormat, ctx)
	require.NoError(t, err)
	frames, _, _, err := stream.Push([]byte(redactedStream))
	require.NoError(t, err)
	tail, _, _, err := stream.Finalize(nil)
	require.NoError(t, err)
	output := string(bytes.Join(append(frames, tail...), nil))
	for _, marker := range []string{`"signature":"signed"`, `"data":"opaque+/="`, `"type":"redacted_thinking"`, `"text":"answer"`} {
		_, after, ok := strings.Cut(output, marker)
		require.True(t, ok, "missing %s", marker)
		output = after
	}
	reply, _, err := engine.DecodeResponseStream(messagesFormat, []byte(redactedStream), ctx)
	require.NoError(t, err)
	result, err := engine.EncodeResponse(messagesFormat, reply, llmprotocol.Envelope{})
	require.NoError(t, err)
	var wire anthropicResponseWire
	require.NoError(t, json.Unmarshal(result.Body, &wire))
	require.JSONEq(t, redactedBlocks, string(wire.Content))
}

func TestAnthropicRedactedReplay(t *testing.T) {
	engine := NewBuiltinEngine()
	reply, _, _, err := engine.DecodeResponse(messagesFormat, []byte(redactedResponse))
	require.NoError(t, err)
	req, env, _, err := engine.DecodeRequest(messagesFormat, []byte(`{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"start"}]}`))
	require.NoError(t, err)
	req.Generation++
	req.Messages = append(req.Messages, llmprotocol.Message{Role: reply.Output[0].Role, Content: reply.Output[0].Content}, req.Messages[0])
	result, err := engine.EncodeRequest(messagesFormat, req, env)
	require.NoError(t, err)
	var wire anthropicRequestWire
	require.NoError(t, json.Unmarshal(result.Body, &wire))
	require.JSONEq(t, redactedBlocks, string(wire.Messages[1].Content))
}

func TestAnthropicRedactedBoundaries(t *testing.T) {
	engine := NewBuiltinEngine()
	plain := strings.Replace(redactedRequest, `{"type":"redacted_thinking","data":"opaque+/="},`, "", 1)
	result, err := engine.TranslateRequest(messagesFormat, messagesFormat, []byte(plain), func(r *llmprotocol.Request) error { r.Model = "backend"; return nil })
	require.NoError(t, err)
	require.Equal(t, `{"model":"backend","messages":[{"role":"assistant","content":[{"signature":"signed","thinking":"thought","type":"thinking"},{"text":"answer","type":"text"}]},{"role":"user","content":[{"text":"next","type":"text"}]}],"max_tokens":16}`, string(result.Body))
	_, _, _, err = engine.DecodeRequest(messagesFormat, []byte(strings.Replace(redactedRequest, `,"data":"opaque+/="`, "", 1)))
	assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_content_variant")
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIImagesV1} {
		_, err := engine.TranslateRequest(messagesFormat, target, []byte(redactedRequest), nil)
		assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "redacted_reasoning")
		_, err = engine.TranslateResponse(messagesFormat, target, []byte(redactedResponse), nil)
		assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "redacted_reasoning")
	}
}
