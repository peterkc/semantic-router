package protocolcodec

import (
	"encoding/json"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func decodeResponsesPhase(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}
	var phase string
	err := json.Unmarshal(raw, &phase)
	return phase, err
}

func encodeResponsesPhase(phase string) json.RawMessage {
	if phase == "" {
		return nil
	}
	raw, _ := json.Marshal(phase)
	return raw
}

func responsesRequestPhaseOmissions(request llmprotocol.Request, target llmprotocol.WireFormat) llmprotocol.Diagnostics {
	if target == llmprotocol.OpenAIResponsesV1 || !slices.ContainsFunc(request.Messages, func(message llmprotocol.Message) bool { return message.ResponsesPhase != "" }) {
		return nil
	}
	return responsesPhaseOmission(target, "input[].phase")
}

func responsesOutputPhaseOmissions(response llmprotocol.Response, target llmprotocol.WireFormat) llmprotocol.Diagnostics {
	if target == llmprotocol.OpenAIResponsesV1 || !slices.ContainsFunc(response.Output, func(item llmprotocol.OutputItem) bool { return item.ResponsesPhase != "" }) {
		return nil
	}
	return responsesPhaseOmission(target, "output.phase")
}

func responsesPhaseOmission(target llmprotocol.WireFormat, field string) llmprotocol.Diagnostics {
	return llmprotocol.Diagnostics{{
		Source: llmprotocol.OpenAIResponsesV1, Target: target, Field: field,
		Action: llmprotocol.DiagnosticDropped, Reason: "assistant message phase is only available in Responses",
	}}
}
