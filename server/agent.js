import { ChatGroq } from "@langchain/groq";
import { StateGraph, MessagesAnnotation, MemorySaver } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { searchKnowledgeBase } from "./tools.js";

const checkpointer = new MemorySaver();
const tools = [searchKnowledgeBase];
const toolNode = new ToolNode(tools);

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  apiKey: process.env.GROQ_API_KEY,
  temperature: 0,
}).bindTools(tools);

function shouldContinue({ messages }) {
  const last = messages[messages.length - 1];
  return last.tool_calls?.length ? "tools" : "__end__";
}

async function callModel({ messages }) {
  const systemMsg = {
    role: "system",
    content: `You are a helpful AI assistant with access to a knowledge base. 
    When users ask questions, search the knowledge base using the available tools. 
    Be concise and accurate.`,
  };
  const response = await model.invoke([systemMsg, ...messages]);
  return { messages: [response] };
}

const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge("__start__", "agent")
  .addConditionalEdges("agent", shouldContinue)
  .addEdge("tools", "agent")
  .compile({ checkpointer });

export async function runAgent({ sessionId = "default", message }) {
  try {
    console.log(`🤖 Running agent for: "${message}"`);

    const response = await graph.invoke(
      { messages: [{ role: "user", content: message }] },
      { configurable: { thread_id: sessionId } }
    );

    const last = response.messages[response.messages.length - 1];
    const output = typeof last.content === "string"
      ? last.content
      : Array.isArray(last.content)
        ? last.content.map((p) => (typeof p === "string" ? p : p?.text ?? "")).join("")
        : String(last.content ?? "");

    console.log(`✅ Agent response success!`);
    return { output };
  } catch (error) {
    console.error("❌ Error in runAgent:", error);
    throw error;
  }
}