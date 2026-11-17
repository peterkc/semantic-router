package extproc

import (
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// isOpenAIHostedResponses uses provider identity and the resolved service URL.
// Compatibility gateways do not establish the provenance of opaque state.
func isOpenAIHostedResponses(profile *config.ProviderProfile, endpoint string, target llmprotocol.WireFormat) bool {
	if target != llmprotocol.OpenAIResponsesV1 || profile == nil || profile.Type != "openai" {
		return false
	}
	if profile.BaseURL != "" {
		endpoint = profile.BaseURL
	}
	if !strings.Contains(endpoint, "://") {
		endpoint = "https://" + endpoint
	}
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return false
	}
	host := strings.ToLower(parsed.Hostname())
	return host == "api.openai.com" || host == "chatgpt.com" || strings.HasSuffix(host, ".openai.azure.com")
}

func allowsResponsesEncryptedRelay(profile *config.ProviderProfile, endpoint string, target llmprotocol.WireFormat) bool {
	return target == llmprotocol.OpenAIResponsesV1 && profile != nil &&
		(profile.AllowEncryptedReasoningRelay || isOpenAIHostedResponses(profile, endpoint, target))
}

// Every endpoint must qualify because Envoy can send the model to any member
// of its pool. A shadow dispatch instead names one concrete endpoint.
func (r *OpenAIRouter) projectResponsesReasoningForBackend(request llmprotocol.Request, model string, target llmprotocol.WireFormat, shadows ...*shadowTarget) (llmprotocol.Request, llmprotocol.Diagnostics) {
	allowRelay := false
	if len(shadows) > 0 {
		shadow := shadows[0]
		allowRelay = shadow != nil && allowsResponsesEncryptedRelay(shadow.profile, shadow.baseURL, target)
	} else if r != nil && r.Config != nil {
		endpoints := r.Config.GetEndpointsForModel(model)
		allowRelay = len(endpoints) > 0
		for _, endpoint := range endpoints {
			profile, profileErr := r.Config.GetProviderProfileForEndpoint(endpoint.Name)
			address, addressErr := endpoint.ResolveAddress(r.Config.ProviderProfiles)
			if profileErr != nil || addressErr != nil || !allowsResponsesEncryptedRelay(profile, address, target) {
				allowRelay = false
				break
			}
		}
	}
	return llmprotocol.ProjectResponsesReasoning(request, target, allowRelay)
}
