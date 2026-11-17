package protocolcodec

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const primeResponsesPhaseItem = `{"type":"message","id":"msg_1","role":"assistant","status":"completed","phase":"final_answer","content":[{"type":"output_text","text":"done"}]}`

func TestPrimeResponsesRequestPhase(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"public","input":[` + primeResponsesPhaseItem + `,{"role":"user","content":"next"}]}`)
	result, err := engine.TranslateRequest(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, body, func(request *llmprotocol.Request) error {
		request.Model = "upstream"
		return nil
	})
	require.NoError(t, err)
	require.Contains(t, string(result.Body), `"model":"upstream"`)
	require.Contains(t, string(result.Body), `"phase":"final_answer"`)
}

func TestPrimeResponsesBufferedPhase(t *testing.T) {
	body := []byte(`{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"upstream","output":[` + primeResponsesPhaseItem + `],"error":null,"incomplete_details":null}`)
	result, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, body, func(response *llmprotocol.Response) error {
		response.Model = "public"
		return nil
	})
	require.NoError(t, err)
	for _, diagnostic := range result.Diagnostics {
		require.NotEqual(t, "output.phase", diagnostic.Field)
	}
	require.Contains(t, string(result.Body), `"model":"public"`)
	require.Contains(t, string(result.Body), `"phase":"final_answer"`)
}

func TestPrimeResponsesStreamPhase(t *testing.T) {
	frames := responsesReasoningFrames(t)
	for _, frame := range frames {
		if item, ok := frame["item"].(map[string]any); ok && item["type"] == "message" {
			item["phase"] = "commentary"
		}
		if frame["type"] == "response.completed" {
			output := frame["response"].(map[string]any)["output"].([]any)
			output[1].(map[string]any)["phase"] = "commentary"
		}
	}
	events, err := decodeResponsesReasoningFrames(t, frames)
	require.NoError(t, err)
	encoder := (OpenAIResponsesCodec{}).NewEncoder(llmprotocol.StreamContext{Context: t.Context()}, llmprotocol.DefaultPolicy())
	var stream strings.Builder
	for _, event := range events {
		encoded, _, err := encoder.Push(event)
		require.NoError(t, err)
		for _, frame := range encoded {
			stream.Write(frame)
		}
	}
	var phases []string
	for _, line := range strings.Split(stream.String(), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var frame struct {
			Type     string                       `json:"type"`
			Item     struct{ Type, Phase string } `json:"item"`
			Response struct {
				Output []struct{ Type, Phase string } `json:"output"`
			} `json:"response"`
		}
		require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &frame))
		if frame.Item.Type == "message" {
			phases = append(phases, frame.Type+":"+frame.Item.Phase)
		}
		if frame.Type == "response.completed" {
			require.Len(t, frame.Response.Output, 2)
			phases = append(phases, frame.Type+":"+frame.Response.Output[1].Phase)
		}
	}
	require.Equal(t, []string{"response.output_item.added:commentary", "response.output_item.done:commentary", "response.completed:commentary"}, phases)
}

func TestResponsesPhaseRejectsWrongRoleOrType(t *testing.T) {
	for _, item := range []string{
		`{"type":"message","role":"assistant","phase":true,"content":"bad"}`,
		`{"type":"reasoning","phase":"commentary","summary":[]}`,
	} {
		_, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(`{"model":"m","input":[`+item+`]}`))
		assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "")
	}
}
