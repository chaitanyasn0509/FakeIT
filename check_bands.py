import rasterio

# CHANGE THIS PATH TO ONE OF YOUR .tif FILES
image_path = r"C:\Users\Chaitanya\Downloads\train\clearDNclips\10m\R141_T13TEE\7.tif"

with rasterio.open(image_path) as src:
    print("=" * 40)
    print("File:", image_path)
    print("Number of bands:", src.count)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Data type:", src.dtypes)
    print("CRS:", src.crs)
    print("=" * 40)