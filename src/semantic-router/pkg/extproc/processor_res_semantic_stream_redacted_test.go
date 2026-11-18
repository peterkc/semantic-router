package extproc

import (
	"github.com/stretchr/testify/require"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"testing"
)

func TestSemanticStreamKeepsRedactedThinking(t *testing.T) {
	state := semanticResponseStreamState{items: make(map[int]*semanticStreamItem)}
	block := llmprotocol.Content{Kind: llmprotocol.ContentRedactedThinking, Data: "opaque"}
	state.observe([]llmprotocol.Event{{Type: llmprotocol.EventOutputItemStarted, Content: &block}, {Type: llmprotocol.EventOutputItemCompleted}, {Type: llmprotocol.EventResponseCompleted}})
	reply, err := state.response()
	require.NoError(t, err)
	require.Equal(t, []llmprotocol.Content{block}, reply.Output[0].Content)
}
