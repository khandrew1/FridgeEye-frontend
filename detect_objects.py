from gradio_client import Client, handle_file

client = Client("iashin/YOLOv3")
result = client.predict(
		source_img=handle_file('InsideFridge6.jpg'),
		api_name="/predict"
)
print(result)