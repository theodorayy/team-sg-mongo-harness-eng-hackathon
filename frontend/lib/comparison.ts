import type { Context, DashboardData, Message, ResponseResult } from "./types";

export class UpstreamError extends Error {
  constructor(
    message: string,
    public status = 502,
  ) {
    super(message);
  }
}

export class MemoryClient {
  constructor(
    private baseUrl: string,
    private fetcher: typeof fetch = fetch,
    private apiToken?: string,
  ) {}
  async request(method: string, path: string, body?: unknown) {
    const response = await this.fetcher(
      `${this.baseUrl.replace(/\/$/, "")}${path}`,
      {
        method,
        headers: {
          "Content-Type": "application/json",
          ...(this.apiToken ? { Authorization: `Bearer ${this.apiToken}` } : {}),
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: AbortSignal.timeout(30_000),
        cache: "no-store",
      },
    );
    const data = await response.json();
    if (!response.ok) {
      console.error(`Memory API error: ${method} ${path} → ${response.status}`, data);
      throw new UpstreamError(
        `Memory API [${method} ${path}]: ${data.error || `HTTP ${response.status}`}`,
        response.status,
      );
    }
    return data;
  }
}

export type ComparisonInput = {
  prompt: string;
  session_id: string;
  request_id: string;
};
export type ComparisonConfig = {
  apiUrl: string;
  memoryApiToken?: string;
  apiKey: string;
  model: string;
  fetcher?: typeof fetch;
};

const baselineInstructions =
  "You are an assistant with access to a NYC 311 complaint dataset. Use the raw complaint records provided below to answer the user's question directly. Be specific with locations, complaint types, and frequencies.";

const memoryInstructions =
  "You are an assistant with access to a knowledge graph built from 2 million NYC 311 complaint records. Use the retrieved knowledge graph below to answer the user's question directly and assertively. Cite source IDs where relevant. Do not add disclaimers about data limitations or suggest the user needs additional data.";

function readContext(value: unknown): Context {
  const context = value as Context;
  if (
    !context ||
    !Array.isArray(context.nodes) ||
    !Array.isArray(context.edges) ||
    !Array.isArray(context.source_ids) ||
    !Array.isArray(context.seed_ids) ||
    typeof context.context_text !== "string" ||
    typeof context.token_count !== "number" ||
    typeof context.truncated !== "boolean" ||
    !Number.isFinite(context.token_count) ||
    context.token_count < 0 ||
    !context.source_ids.every((id) => typeof id === "string") ||
    !context.nodes.every(
      (node) =>
        typeof node.id === "string" &&
        typeof node.text === "string" &&
        Array.isArray(node.source_ids),
    ) ||
    !context.edges.every(
      (edge) =>
        typeof edge.source_id === "string" &&
        typeof edge.target_id === "string",
    )
  ) {
    throw new UpstreamError(
      "The memory API returned an invalid retrieval payload.",
    );
  }
  return context;
}

export async function compareResponses(
  input: ComparisonInput,
  config: ComparisonConfig,
): Promise<DashboardData> {
  const fetcher = config.fetcher || fetch;
  const client = new MemoryClient(config.apiUrl, fetcher, config.memoryApiToken);
  const path = `/v1/sessions/${encodeURIComponent(input.session_id)}`;
  const baselinePath = `/v1/sessions/${encodeURIComponent(`${input.session_id}.baseline`)}`;
  const { messages } = await client.request("GET", `${path}/messages?limit=20`);
  if (!Array.isArray(messages))
    throw new UpstreamError(
      "The memory API returned invalid conversation history.",
    );
  const history = (messages as Message[])
    .filter(
      (m) =>
        ["user", "assistant"].includes(m.role) && typeof m.content === "string",
    )
    .map(({ role, content }) => ({ role, content }));
  const [turn, baselineTurn] = await Promise.all([
    client.request("POST", `${path}/turns`, {
      prompt: input.prompt,
      idempotency_key: input.request_id,
    }),
    client.request("POST", `${baselinePath}/turns`, {
      prompt: input.prompt,
      idempotency_key: input.request_id,
    }),
  ]);
  if (
    typeof turn.turn_id !== "string" ||
    typeof baselineTurn.turn_id !== "string"
  )
    throw new UpstreamError("The memory API returned an invalid turn.");
  const retrievalStart = performance.now();
  const context = readContext(
    await client.request(
      "POST",
      `${path}/turns/${encodeURIComponent(turn.turn_id)}/context`,
      {
        retrieval_limits: {
          seed_limit: 5,
          max_hops: 2,
          max_nodes: 30,
          max_edges: 60,
          max_context_tokens: 2000,
        },
      },
    ),
  );
  const retrieval_ms = Math.round(performance.now() - retrievalStart);
  async function generate(
    contextText: string,
    useMemoryInstructions: boolean,
  ): Promise<ResponseResult> {
    const start = performance.now();
    const instructions = useMemoryInstructions
      ? memoryInstructions
      : baselineInstructions;
    const systemContent = contextText
      ? `${instructions}\n\n${useMemoryInstructions ? "Retrieved knowledge graph" : "Raw dataset records"}:\n${contextText}`
      : instructions;
    const response = await fetcher(
      "https://openrouter.ai/api/v1/chat/completions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${config.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: config.model,
          temperature: 0,
          max_tokens: 700,
          messages: [
            {
              role: "system",
              content: systemContent,
            },
            ...history,
            { role: "user", content: input.prompt },
          ],
        }),
        signal: AbortSignal.timeout(90_000),
      },
    );
    if (!response.ok)
      throw new UpstreamError(
        `Model request failed (HTTP ${response.status}). Check the server's model configuration.`,
        502,
      );
    const result = await response.json();
    const text = result.choices?.[0]?.message?.content;
    if (typeof text !== "string" || !text.trim())
      throw new UpstreamError("The model returned an empty response.");
    return {
      text,
      current_source_ids: [],
      historical_source_ids: [],
      latency_ms: Math.round(performance.now() - start),
      input_tokens: result.usage?.prompt_tokens ?? null,
      output_tokens: result.usage?.completion_tokens ?? null,
    };
  }
  const rawSources = await client.request("POST", "/v1/sources/search", {
    query: input.prompt,
    limit: 200,
  });
  const baselineContext =
    typeof rawSources.text === "string" && rawSources.text
      ? rawSources.text
      : context.context_text;
  const [baseline, memory] = await Promise.all([
    generate(baselineContext, false),
    generate(context.context_text, true),
  ]);
  memory.historical_source_ids = [...new Set(context.source_ids)];
  await Promise.all([
    client.request(
      "POST",
      `${path}/turns/${encodeURIComponent(turn.turn_id)}/response`,
      { content: memory.text, idempotency_key: `${turn.turn_id}-assistant` },
    ),
    client.request(
      "POST",
      `${baselinePath}/turns/${encodeURIComponent(baselineTurn.turn_id)}/response`,
      {
        content: baseline.text,
        idempotency_key: `${baselineTurn.turn_id}-assistant`,
      },
    ),
  ]);
  return {
    mode: "live",
    generated_at: new Date().toISOString(),
    adapter: config.model,
    record_count: context.source_ids.length,
    batch_size: 0,
    batches: [],
    records: [],
    nodes: context.nodes,
    edges: context.edges,
    baseline,
    memory,
    context,
    prompt: input.prompt,
    retrieval_ms,
  };
}
