package train

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// loadTrainText reads plain text or messages JSONL (flattened for GPT-2 smoke).
func loadTrainText(path string) (string, error) {
	if strings.EqualFold(filepath.Ext(path), ".jsonl") {
		return MessagesToText(path)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// Message is one chat turn in TRL-style JSONL.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// MessagesExample is one JSONL row.
type MessagesExample struct {
	Messages []Message `json:"messages"`
}

// MessagesToText flattens messages JSONL into plain text for GPT-2 LoRA smoke.
// Not a correct chat template — use ApplyChatTemplate for instruct formatting.
func MessagesToText(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	var parts []string
	for {
		var ex MessagesExample
		if err := dec.Decode(&ex); err != nil {
			if err == io.EOF {
				break
			}
			return "", err
		}
		for _, m := range ex.Messages {
			if m.Role == "user" || m.Role == "assistant" {
				parts = append(parts, m.Content)
			}
		}
	}
	if len(parts) == 0 {
		return "", fmt.Errorf("train: no user/assistant messages in %s", path)
	}
	return strings.Join(parts, "\n\n"), nil
}

// ValidateMessagesJSONL checks TRL-style messages rows.
func ValidateMessagesJSONL(path string) (total int, issues []string, err error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	line := 0
	for {
		line++
		var obj map[string]any
		if err := dec.Decode(&obj); err != nil {
			if err == io.EOF {
				break
			}
			issues = append(issues, fmt.Sprintf("line %d: invalid json (%v)", line, err))
			continue
		}
		total++
		msgs, ok := obj["messages"].([]any)
		if !ok || len(msgs) < 2 {
			issues = append(issues, fmt.Sprintf("line %d: messages must be a list with >=2 items", line))
			continue
		}
		roles := map[string]bool{}
		for _, m := range msgs {
			mm, ok := m.(map[string]any)
			if !ok {
				issues = append(issues, fmt.Sprintf("line %d: message not object", line))
				break
			}
			role, _ := mm["role"].(string)
			_, hasContent := mm["content"]
			if role == "" || !hasContent {
				issues = append(issues, fmt.Sprintf("line %d: message missing role/content", line))
				break
			}
			if role != "system" && role != "user" && role != "assistant" {
				issues = append(issues, fmt.Sprintf("line %d: bad role %s", line, role))
			}
			roles[role] = true
		}
		if !roles["user"] || !roles["assistant"] {
			issues = append(issues, fmt.Sprintf("line %d: need user and assistant roles", line))
		}
	}
	return total, issues, nil
}
