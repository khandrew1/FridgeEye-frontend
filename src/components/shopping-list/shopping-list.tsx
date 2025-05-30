"use client";

import { useState, useEffect } from "react";
import { ref, onValue } from "firebase/database"; 
import { db } from "@/firebase"; 
import ShoppingCard from "./shopping-card";

interface ShoppingItem {
  id: string;
  itemName: string;
  image: string;
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
            image: data[key].imageBase64,
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

  if (isLoading) {
    return <div className="flex justify-center items-center h-full w-full"><p>Loading shopping list...</p></div>;
  }

  if (error) {
    return <div className="flex justify-center items-center h-full w-full"><p className="text-red-500">{error}</p></div>;
  }

  return (
    <div className="p-4 w-full">
      {items.length === 0 ? (
        <p className="text-center text-gray-500">Your shopping list is empty. Add some items!</p>
      ) : (
        <div className="grid grid-cols-1 overflow-y-scroll sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {items.map((item) => (
            <ShoppingCard
              key={item.id}
              itemName={item.itemName}
              image={item.image}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default ShoppingList;
