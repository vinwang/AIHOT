// The engine's own model: one OpenAI-compatible model (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL) runs every
// step of the analysis when nothing picks another one for a step.
import { Reply, stub, tag } from "./setup.ts";
import { analysisStep, SELECTING_SCORE, type AnalysisStep } from "./analysis-steps.ts";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { after, before, test } from "node:test";
import sharp from "sharp";
import { config } from "@aihot/backend/config";
import { closeDb, sql } from "@aihot/backend/db";
import { upsertMaterial } from "@aihot/backend/content/materials";
import { analyzeArticle, loadAnalyzeInput, runStructure, runUnderstand, UNDERSTAND_FLOOR } from "@aihot/backend/editorial/analyze";
import { CAPABILITIES, type Capability } from "@aihot/backend/editorial/models";
import { stopBoss } from "@aihot/backend/jobs/queue";

// Nothing chosen per step: every capability falls back to the `default` model (a step the site
// gives a model of its own is sent back to it by its environment variable).
for (const c of Object.values(CAPABILITIES) as Capability[]) {
  if (c.default === "default") delete process.env[c.env];
  else process.env[c.env] = "default";
}

const T = tag();
const SOURCE = `test-default-model-${T}`;
const seen: Array<{ model: string; step: AnalysisStep; content: unknown }> = [];
let failCombined = false;
const provider = await stub((_hit, req) => {
  const body = JSON.parse(req.body) as { model: string; messages: Array<{ content: unknown }> };
  const step = analysisStep(req.body);
  const content = body.messages.at(-1)!.content;
  const user = String(content);
  seen.push({ model: body.model, step, content });
  if (step === "structure-understand" && failCombined) {
    failCombined = false;
    return new Reply(503, { error: "temporary failure" });
  }
  if ((step === "structure-understand" && (user.includes("CONTENT_FILTER") || user.includes("WRITING_FILTER")))
    || (step === "understand" && user.includes("WRITING_FILTER"))) {
    return new Reply(400, { contentFilter: [{ level: 1 }], error: { code: "1301", message: "refused" } });
  }
  const structure = { category: "ai-products", tags: ["产品更新"], subjects: [], scope: "single", fact: null };
  const writing = { itemType: "product_launch", authorRole: "principal", tags: ["产品更新"], editorialJudgment: "理由", titleZh: "一个模型的标题", summaryZh: "一个模型写的摘要。第二句。" };
  const answer =
    step === "prefilter" ? { label: "PASS", reason: "测试" }
    : step === "score" ? { attentionScore: user.includes("LOW_SCORE") ? Math.max(0, UNDERSTAND_FLOOR - 1) : SELECTING_SCORE }
    : step === "understand" ? writing
    : step === "structure" ? structure
    : step === "structure-understand" ? { structure, writing }
    : "title_zh: 标题\nsummary_zh: 摘要。";
  return { id: `stub-${seen.length}`, choices: [{ message: { content: typeof answer === "string" ? answer : JSON.stringify(answer) } }], usage: { prompt_tokens: 1, completion_tokens: 1 } };
});
Object.assign(process.env, { LLM_BASE_URL: `${provider.url}/v1`, LLM_API_KEY: "test-key", LLM_MODEL: "one-model", MODEL_CALLS_ENABLED: "true" });

before(async () => {
  await sql`INSERT INTO sources (id, name, kind, tier, participation_mode, next_fetch_at) VALUES (${SOURCE}, 'Test default model', 'rss', 'T1', 'editorial', '2100-01-01')`;
});
after(async () => {
  await provider.close();
  await stopBoss();
  await closeDb();
});

async function article(marker: string) {
  const { articleId } = await upsertMaterial({
    sourceId: SOURCE, url: `https://example.com/${marker}-${T}`, title: `A product launch ${marker} ${T}`,
    bodyText: `A company launched a product with pricing and availability. ${marker} ${T} `.repeat(6),
    bodyStatus: "ok", via: "fetch", publishedAt: new Date(),
  } as never);
  return articleId;
}

test("one model keeps both scores and combines structure and writing in one request", async () => {
  const articleId = await article("COMBINED");
  const res = await analyzeArticle(articleId);
  assert.equal(res!.output!.selected, true);
  assert.equal(res!.output!.titleZh, "一个模型的标题");
  assert.deepEqual([res!.output!.score, res!.output!.category, res!.output!.scope], [SELECTING_SCORE, "ai-products", "single"]);
  assert.deepEqual(seen.map((r) => r.step).sort(), ["prefilter", "score", "score", "structure-understand"]);
  assert.ok(seen.every((r) => r.model === "one-model"), "every request names the configured model");
  assert.equal(res!.receiptIds.length, 4, "the combined request has one receipt");
  assert.equal(new Set(res!.receiptIds).size, 4);
  const services = await sql<{ service: string }[]>`SELECT DISTINCT service FROM receipts WHERE subject LIKE ${`article:${articleId}%`}`;
  assert.deepEqual(services.map((s) => s.service), ["llm"]);
  const calls = seen.length;
  const again = await analyzeArticle(articleId);
  assert.equal(seen.length, calls, "the combined answer is reused");
  assert.deepEqual(again!.receiptIds, res!.receiptIds);
  assert.equal(again!.reused, true);
});

test("a low score keeps the existing summary and structure steps", async () => {
  seen.length = 0;
  const result = await analyzeArticle(await article("LOW_SCORE"));
  assert.equal(result!.output!.selected, false);
  assert.equal(result!.output!.titleZh, "标题");
  assert.equal(result!.output!.category, "ai-products");
  assert.deepEqual(seen.map((r) => r.step).sort(), ["prefilter", "score", "score", "structure", "summarize"]);
});

test("a combined content-filter refusal falls back to the original independent steps", async () => {
  seen.length = 0;
  const articleId = await article("CONTENT_FILTER");
  const result = await analyzeArticle(articleId);
  assert.equal(result!.output!.selected, true);
  assert.equal(result!.output!.titleZh, "一个模型的标题");
  assert.equal(result!.output!.category, "ai-products");
  assert.deepEqual(seen.map((r) => r.step).sort(), ["prefilter", "score", "score", "structure", "structure-understand", "understand"]);
  const calls = seen.length;
  const again = await analyzeArticle(articleId);
  assert.equal(again!.output!.titleZh, result!.output!.titleZh);
  assert.equal(seen.length, calls, "the separate answers are reused");
});

test("a content-filter refusal from writing still falls back to a summary", async () => {
  seen.length = 0;
  const result = await analyzeArticle(await article("WRITING_FILTER"));
  assert.equal(result!.output!.selected, true);
  assert.equal(result!.output!.titleZh, "标题");
  assert.equal(result!.output!.category, "ai-products");
  assert.deepEqual(seen.map((r) => r.step).sort(), ["prefilter", "score", "score", "structure", "structure-understand", "summarize", "understand"]);
});

test("a failed combined request reuses the earlier prefilter and independent scores", async () => {
  seen.length = 0;
  failCombined = true;
  const articleId = await article("RETRY");
  await assert.rejects(analyzeArticle(articleId), /503/);
  const result = await analyzeArticle(articleId);
  assert.equal(result!.output!.selected, true);
  assert.equal(result!.output!.titleZh, "一个模型的标题");
  assert.deepEqual(seen.map((r) => r.step).sort(), ["prefilter", "score", "score", "structure-understand", "structure-understand"]);
});

test("existing separate receipts are reused when an article is reprocessed after an upgrade", async () => {
  const articleId = await article("EXISTING");
  const input = (await loadAnalyzeInput(articleId))!;
  const structure = await runStructure(input);
  const writing = (await runUnderstand(input))!;
  seen.length = 0;
  const result = await analyzeArticle(articleId);
  assert.equal(result!.output!.category, "ai-products");
  assert.equal(result!.output!.titleZh, "一个模型的标题");
  assert.ok(result!.receiptIds.includes(structure.receiptId));
  assert.ok(result!.receiptIds.includes(writing.receiptIds[0]!));
  assert.deepEqual(seen.map((r) => r.step).sort(), ["prefilter", "score", "score"]);
});

test("an unresolved separate receipt is not bypassed by a combined request", async () => {
  const articleId = await article("UNKNOWN");
  const input = (await loadAnalyzeInput(articleId))!;
  const structure = await runStructure(input);
  await sql`UPDATE receipts SET status = 'unknown', response = NULL WHERE id = ${structure.receiptId}`;
  seen.length = 0;
  await assert.rejects(analyzeArticle(articleId), /unknown outcome/);
  assert.ok(!seen.some((r) => r.step === "structure-understand"));
});

test("long material retains the separate structure and writing requests", async () => {
  const articleId = await article("LONG");
  await sql`UPDATE articles SET body_text = ${"A company released a product. ".repeat(1500)} WHERE id = ${articleId}`;
  seen.length = 0;
  const result = await analyzeArticle(articleId);
  assert.equal(result!.output!.selected, true);
  assert.equal(result!.output!.titleZh, "一个模型的标题");
  assert.deepEqual(seen.map((r) => r.step).sort(), ["prefilter", "score", "score", "structure", "understand"]);
});

test("a vision model still receives the image in a separate writing request", async () => {
  const articleId = await article("IMAGE");
  const url = `https://example.com/${T}/image.png`;
  const key = createHash("sha256").update(`thumb|${url}`).digest("hex");
  const file = path.join(config.dataDir, "imgcache", key.slice(0, 2), key);
  const png = await sharp({ create: { width: 2, height: 2, channels: 3, background: "#176b75" } }).png().toBuffer();
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, png);
  await writeFile(`${file}.type`, "image/png");
  await sql`UPDATE articles SET media = ${sql.json([{ kind: "image", url }])} WHERE id = ${articleId}`;
  const vision = process.env.LLM_VISION;
  process.env.LLM_VISION = "true";
  try {
    seen.length = 0;
    const result = await analyzeArticle(articleId);
    assert.equal(result!.output!.titleZh, "一个模型的标题");
    assert.deepEqual(seen.map((r) => r.step).sort(), ["prefilter", "score", "score", "structure", "understand"]);
    const parts = seen.find((r) => r.step === "understand")!.content as Array<{ type: string; image_url?: { url: string } }>;
    assert.equal(parts.find((part) => part.type === "image_url")?.image_url?.url, `data:image/png;base64,${png.toString("base64")}`);
  } finally {
    if (vision === undefined) delete process.env.LLM_VISION;
    else process.env.LLM_VISION = vision;
  }
});
