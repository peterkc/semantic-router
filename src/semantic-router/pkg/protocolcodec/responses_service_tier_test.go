package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestPrimeResponsesRequestServiceTier(t *testing.T) {
	for _, tc := range []struct{ name, value string }{
		{"default", `"default"`},
		{"priority", `"priority"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wire := rebuildResponsesTier(t, `,"service_tier":`+tc.value)
			require.Equal(t, tc.value, string(wire["service_tier"]))
		})
	}
}

func TestResponsesRequestServiceTierOmissionAndType(t *testing.T) {
	for _, suffix := range []string{"", `,"service_tier":null`} {
		require.NotContains(t, rebuildResponsesTier(t, suffix), "service_tier")
	}
	_, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(`{"model":"m","input":"next","service_tier":true}`))
	assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "")
}

func rebuildResponsesTier(t *testing.T, suffix string) map[string]json.RawMessage {
	t.Helper()
	body := []byte(`{"model":"public","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"next"}]}]` + suffix + `}`)
	result, err := NewBuiltinEngine().TranslateRequest(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, body, func(request *llmprotocol.Request) error {
		request.Model = "upstream"
		return nil
	})
	require.NoError(t, err)
	var wire map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(result.Body, &wire))
	require.Equal(t, `"upstream"`, string(wire["model"]))
	return wire
}
