import { ref, push } from "firebase/database";
import { db } from "@/firebase";

export async function POST(request: Request) {
  try {
    const { itemName, quantity } = await request.json();

    const foodItemsRef = ref(db, "foodItems");

    const newItem = {
      itemName,
      quantity,
    };

    await push(foodItemsRef, newItem);

    console.log("Food item added successfully!", newItem);
    return new Response(
      JSON.stringify({ message: "Success!", item: newItem }),
      {
        status: 200,
        headers: {
          "Content-Type": "application/json",
        },
      },
    );
  } catch (err: unknown) {
    console.error("Error adding food item: ", err);

    let errorMessage =
      "Failed to add food item due to an internal server error.";
    let statusCode = 500;

    if (err instanceof SyntaxError) {
      errorMessage = "Invalid JSON payload.";
      statusCode = 400;
    }

    return new Response(JSON.stringify({ message: errorMessage }), {
      status: statusCode,
      headers: {
        "Content-Type": "application/json",
      },
    });
  }
}
