import {
    Card,
    CardContent,
    CardTitle,
} from "@/components/ui/card";
import Image from "next/image";

const ShoppingCard = ({
    itemName,
    image
}: {
    itemName: string;
    image?: string;
}) => {
    const base64URL = `data:image/png;base64,${image}`

    return (
        <Card>
            <div className="flex flex-row px-3">
                <Image src={base64URL} alt="Banana" width={150} height={150} />
                <div className="flex flex-col w-full">
                    <CardContent>
                        <CardTitle>{itemName}</CardTitle>
                    </CardContent>
                </div>
            </div>
        </Card>
    );
};

export default ShoppingCard;
