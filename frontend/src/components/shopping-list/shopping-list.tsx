"use client";

import { useState, useEffect } from "react";
import { ref, onValue } from "firebase/database"; 
import { db } from "@/firebase"; 
import ShoppingCard from "./shopping-card";
import { Button } from "@/components/ui/button";

interface ShoppingItem {
  id: string;
  itemName: string;
  quantity: number;
}

const ShoppingList = () => {
  const [items, setItems] = useState<ShoppingItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const foodItemsRef = ref(db, "foodItems"); 

    // Subscribe to real-time updates
    const unsubscribe = onValue(
      foodItemsRef,
      (snapshot) => {
        const data = snapshot.val();
        if (data) {
          const loadedItems: ShoppingItem[] = Object.keys(data).map((key) => ({
            id: key,
            itemName: data[key].itemName,
            quantity: data[key].quantity,
          }));
          setItems(loadedItems);
        } else {
          setItems([]);
        }
        setIsLoading(false);
      },
      (err) => {
        console.error("Error fetching data:", err);
        setError("Failed to load shopping list. Please try again later.");
        setIsLoading(false);
      }
    );

    // Cleanup subscription on component unmount
    return () => unsubscribe();
  }, []); // Empty dependency array means this effect runs once on mount and cleanup on unmount

  // This function is to add new items, as in your original code
  const handleAddItem = async () => {
    try {
      const response = await fetch("/api/addItem", { // Assuming your API endpoint adds 'createdAt'
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          itemName: "New Item " + Math.floor(Math.random() * 100), // Example dynamic name
          quantity: Math.floor(Math.random() * 10) + 1,
        }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || "Failed to add item");
      }
      const result = await response.json();
      console.log("Item added via API:", result);
      // Realtime listener will update the UI, or you could optimistically update here
    } catch (err) {
      console.error("Error adding item:", err);
      // Optionally, set an error state to show in the UI
    }
  };

  if (isLoading) {
    return <div className="flex justify-center items-center h-full w-full"><p>Loading shopping list...</p></div>;
  }

  if (error) {
    return <div className="flex justify-center items-center h-full w-full"><p className="text-red-500">{error}</p></div>;
  }

  return (
    <div className="p-4 w-full">
      <div className="mb-8 flex justify-center">
        <Button
          onClick={handleAddItem} // Renamed for clarity
          variant="outline"
          size="lg"
          className="cursor-pointer"
        >
          Add Random Item (via API)
        </Button>
      </div>

      {items.length === 0 ? (
        <p className="text-center text-gray-500">Your shopping list is empty. Add some items!</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {items.map((item) => (
            <ShoppingCard
              key={item.id}
              itemName={item.itemName}
              quantity={item.quantity}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default ShoppingList;
