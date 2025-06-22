import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { StateGraph } from "@langchain/langgraph";
import { Annotation } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { MongoClient } from "mongodb";
import { z } from "zod";
import "dotenv/config";

export async function callAgent(
  client: MongoClient,
  query: string,
  thread_id: string
) {
  // Define the MongoDB database and collection
  const dbName = "hr_database";
  const db = client.db(dbName);

  // Define the graph state
  const GraphState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
      reducer: (x, y) => x.concat(y),
    }),
    expenses: Annotation<any[]>({
      reducer: (x, y) => x.concat(y),
    }),
    spendingLimits: Annotation<any>({
      reducer: (x, y) => x.concat(y),
    }),
    spendingCategories: Annotation<any[]>({
      reducer: (x, y) => x.concat(y),
    }),
    alerts: Annotation<any[]>({
      reducer: (x, y) => x.concat(y),
    }),
  });

  const saveSpendingTool = tool(
    async ({ amount }) => {
      console.log("Save spending tool called");

      const spendingsCollection = db.collection("spendings");
      const result = await spendingsCollection.insertOne({
        amount,
        date: new Date(),
      });

      return `Successfully saved your spending of $${amount}.`;
    },
    {
      name: "save_spending",
      description: "Saves the user's spending to the database",
      schema: z.object({
        amount: z.number().describe("The amount spent"),
      }),
    }
  );

  const checkPastWeekSpendingTool = tool(
    async () => {
      console.log("Check past week spending tool called");

      const expensesCollection = db.collection("expenses");
      const oneWeekAgo = new Date();
      oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);

      const expenses = await expensesCollection
        .find({ date: { $gte: oneWeekAgo } })
        .toArray();

      const spendingByCategory = expenses.reduce(
        (acc: { [key: string]: number }, expense) => {
          if (!acc[expense.category]) {
            acc[expense.category] = 0;
          }
          acc[expense.category] += expense.amount;
          return acc;
        },
        {}
      );

      const summary = Object.entries(spendingByCategory)
        .map(([category, amount]) => `You spent $${amount} on ${category}.`)
        .join("\n");

      return `Here's your spending summary for the past week:\n${summary}`;
    },
    {
      name: "check_past_week_spending",
      description: "Checks how much the user has spent in the past week",
      schema: z.object({}),
    }
  );

  const expenseTrackerTool = tool(
    async ({ amount }) => {
      console.log("Expense Tracker Tool called");

      const category = await model.invoke([
        new HumanMessage("What category are you spending on?"),
      ]);

      const expensesCollection = db.collection("expenses");
      const result = await expensesCollection.insertOne({
        amount,
        category: category.content,
        date: new Date(),
      });

      return `Successfully logged your expense of $${amount} in category ${category.content}.`;
    },
    {
      name: "log_expense",
      description: "Logs the user's expense in a specific category",
      schema: z.object({
        amount: z.number().describe("The amount spent"),
      }),
    }
  );

  const tipsRecommendationTool = tool(
    async () => {
      console.log("Tips & Recommendation Tool called");

      const expensesCollection = db.collection("expenses");
      const oneWeekAgo = new Date();
      oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);

      const expenses = await expensesCollection
        .find({ date: { $gte: oneWeekAgo } })
        .toArray();

      const spendingByCategory = expenses.reduce(
        (acc: { [key: string]: number }, expense) => {
          if (!acc[expense.category]) {
            acc[expense.category] = 0;
          }
          acc[expense.category] += expense.amount;
          return acc;
        },
        {}
      );

      const prompt = `Based on the following spending data, provide personalized financial tips:
${Object.entries(spendingByCategory)
  .map(([category, amount]) => `- Spent $${amount} on ${category}`)
  .join("\n")}
`;

      const result = await model.invoke([new HumanMessage(prompt)]);

      return result.content;
    },
    {
      name: "provide_tips",
      description: "Provides advice on the user's spending habits",
      schema: z.object({}),
    }
  );

  const spendingLimitTool = tool(
    async ({ limit }) => {
      console.log("Spending Limit Tool called");

      const spendingLimitsCollection = db.collection("spending_limits");
      const result = await spendingLimitsCollection.insertOne({
        limit,
        date: new Date(),
      });

      return `Successfully set a spending limit of $${limit}.`;
    },
    {
      name: "set_spending_limit",
      description: "Sets a spending limit",
      schema: z.object({
        limit: z.number().describe("The spending limit"),
      }),
    }
  );

  const tools = [
    saveSpendingTool,
    checkPastWeekSpendingTool,
    expenseTrackerTool,
    tipsRecommendationTool,
    spendingLimitTool,
  ];

  // We can extract the state typing via `GraphState.State`
  const toolNode = new ToolNode<typeof GraphState.State>(tools);

  const model = new ChatOpenAI({
    modelName: "gpt-4o",
    temperature: 1,
  }).bindTools(tools);

  // Define the function that determines whether to continue or not
  function shouldContinue(state: typeof GraphState.State) {
    const messages = state.messages;
    const lastMessage = messages[messages.length - 1] as AIMessage;

    // If the LLM makes a tool call, then we route to the "tools" node
    if (lastMessage.tool_calls?.length) {
      return "tools";
    }
    // Otherwise, we stop (reply to the user)
    return "__end__";
  }

  // Define the function that calls the model
  async function callModel(state: typeof GraphState.State) {
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a helpful AI finance assistant, collaborating with other specialized financial assistants. Use the provided tools to help users track their expenses, manage budgets, and receive personalized financial advice. If you are unable to fully answer, that's OK — another assistant with different capabilities will continue where you left off. Execute whatever tasks you can to move the user toward financial clarity and actionable insights. If you or any of the other assistants arrive at a complete recommendation or report, prefix your response with FINAL ANSWER so the team knows to stop. You have access to the following tools: {tool_names}.
{system_message}
Current time: {time}.
`,
      ],
      new MessagesPlaceholder("messages"),
    ]);

    const formattedPrompt = await prompt.formatMessages({
      system_message: "You are helpful Financial assistant Chatbot Agent.",
      time: new Date().toISOString(),
      tool_names: tools.map((tool) => tool.name).join(", "),
      messages: state.messages,
    });

    const result = await model.invoke(formattedPrompt);

    return { messages: [result] };
  }

  // Define a new graph
  const workflow = new StateGraph(GraphState)
    .addNode("agent", callModel)
    .addNode("tools", toolNode)
    .addEdge("__start__", "agent")
    .addConditionalEdges("agent", shouldContinue)
    .addEdge("tools", "agent");

  // Initialize the MongoDB memory to persist state between graph runs
  const checkpointer = new MongoDBSaver({ client, dbName });

  // This compiles it into a LangChain Runnable.
  // Note that we're passing the memory when compiling the graph
  const app = workflow.compile({ checkpointer });

  // Use the Runnable
  const finalState = await app.invoke(
    {
      messages: [new HumanMessage(query)],
    },
    { recursionLimit: 15, configurable: { thread_id: thread_id } }
  );

  // console.log(JSON.stringify(finalState.messages, null, 2));
  console.log(finalState.messages[finalState.messages.length - 1].content);

  return finalState.messages[finalState.messages.length - 1].content;
}
