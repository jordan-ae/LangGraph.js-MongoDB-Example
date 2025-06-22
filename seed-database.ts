import { MongoClient } from "mongodb";
import "dotenv/config";

const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string);

async function seedDatabase(): Promise<void> {
  try {
    await client.connect();
    await client.db("admin").command({ ping: 1 });
    console.log("Pinged your deployment. You successfully connected to MongoDB!");

    const db = client.db("hr_database");
    const expensesCollection = db.collection("expenses");

    // Clear existing data
    await expensesCollection.deleteMany({});

    // Seed expenses
    const expenses = [
      { amount: 100, category: "Food", date: new Date() },
      { amount: 200, category: "Transport", date: new Date() },
      { amount: 50, category: "Entertainment", date: new Date() },
    ];

    await expensesCollection.insertMany(expenses);
    console.log("Successfully seeded expenses");

    console.log("Database seeding completed");

  } catch (error) {
    console.error("Error seeding database:", error);
  } finally {
    await client.close();
  }
}

seedDatabase().catch(console.error);
