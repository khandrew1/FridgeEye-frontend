import {
  Card,
  CardContent,
  CardDescription,
  CardTitle,
} from "@/components/ui/card";
import Image from "next/image";

const ShoppingCard = ({
  itemName,
  quantity,
}: {
  itemName: string;
  quantity: number;
}) => {
  return (
    <Card>
      <div className="flex flex-row px-3">
        <Image src="/banana.webp" alt="Banana" width={150} height={150} />
        <div className="flex flex-col w-full">
          <CardContent>
            <CardTitle>{itemName}</CardTitle>
            <CardDescription>Quantity: {quantity}</CardDescription>
          </CardContent>
        </div>
      </div>
    </Card>
  );
};

export default ShoppingCard;
