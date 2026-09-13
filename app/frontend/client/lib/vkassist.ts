export type Citation = {
  id: number;
  document: string;
  page_start: number;
  page_end: number;
  section: string | null;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  error?: boolean;
};

type AnswerResponse = {
  answer: string;
  citations: Citation[];
};

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export const sourceDocumentUrl = `${API_BASE_URL}/v1/documents/vulkan-documentation`;
export const projectReadmeUrl = `${API_BASE_URL}/v1/documents/project-readme`;

export async function chatWithVkAssist(prompt: string): Promise<ChatMessage> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/v1/answers`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: prompt,
        limit: 5,
        rerank: true,
        context_window: 1,
        parent_context: false,
      }),
    });
  } catch {
    throw new Error("VkAssist API is unreachable. Start FastAPI on port 8000 and try again.");
  }

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? "VkAssist could not generate an answer. Please try again.");
  }

  const data = (await response.json()) as AnswerResponse;
  return {
    id: `assistant-${Date.now()}`,
    role: "assistant",
    content: data.answer,
    citations: data.citations,
  };
}
