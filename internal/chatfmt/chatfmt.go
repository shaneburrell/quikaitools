package chatfmt

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Message is a chat turn.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Apply applies a named chat template to messages (plus optional trailing user prompt).
// Supported: chatml (Qwen/ChatML), raw (concat).
func Apply(template string, messages []Message, prompt string) (string, error) {
	msgs := append([]Message(nil), messages...)
	if strings.TrimSpace(prompt) != "" {
		msgs = append(msgs, Message{Role: "user", Content: prompt})
	}
	if len(msgs) == 0 {
		return "", fmt.Errorf("chatfmt: no messages")
	}
	switch strings.ToLower(strings.TrimSpace(template)) {
	case "", "chatml", "qwen":
		return chatML(msgs), nil
	case "raw", "none":
		return rawConcat(msgs), nil
	default:
		return "", fmt.Errorf("chatfmt: unknown template %q (want chatml|raw)", template)
	}
}

// ParseMessagesJSON parses a JSON array of {role,content}.
func ParseMessagesJSON(s string) ([]Message, error) {
	var msgs []Message
	if err := json.Unmarshal([]byte(s), &msgs); err != nil {
		return nil, fmt.Errorf("chatfmt: messages json: %w", err)
	}
	return msgs, nil
}

func chatML(msgs []Message) string {
	var b strings.Builder
	for _, m := range msgs {
		role := m.Role
		if role == "" {
			role = "user"
		}
		fmt.Fprintf(&b, "<|im_start|>%s\n%s<|im_end|>\n", role, m.Content)
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}

func rawConcat(msgs []Message) string {
	var parts []string
	for _, m := range msgs {
		parts = append(parts, m.Content)
	}
	return strings.Join(parts, "\n")
}
