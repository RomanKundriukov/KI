using NumSharp;
using SkiaSharp;

namespace KI_TensorFlow_Keras.Services
{
    public class ImageLoader
    {
        public static NDArray LoadImagesFromDirectory(string directoryPath, int imageHeight, int imageWidth, int batchSize)
        {
            var imageFiles = Directory.GetFiles(directoryPath, "*.jpg");
            var numImages = imageFiles.Length;
            var dataset = np.zeros((numImages, imageHeight, imageWidth, 3), np.float32); // Массив для хранения изображений

            for (int i = 0; i < numImages; i++)
            {
                string imagePath = imageFiles[i];
                using (var inputStream = File.OpenRead(imagePath))
                using (var skBitmap = SKBitmap.Decode(inputStream))
                {
                    // Изменение размера изображения
                    using (var resizedBitmap = skBitmap.Resize(new SKImageInfo(imageWidth, imageHeight), SKFilterQuality.High))
                    {
                        if (resizedBitmap == null)
                            continue;

                        var pixels = new float[imageHeight * imageWidth * 3];
                        int pixelIndex = 0;

                        for (int y = 0; y < resizedBitmap.Height; y++)
                        {
                            for (int x = 0; x < resizedBitmap.Width; x++)
                            {
                                var color = resizedBitmap.GetPixel(x, y);
                                pixels[pixelIndex++] = color.Red / 255.0f;
                                pixels[pixelIndex++] = color.Green / 255.0f;
                                pixels[pixelIndex++] = color.Blue / 255.0f;
                            }
                        }
                        dataset[i] = np.array(pixels).reshape(imageHeight, imageWidth, 3); // Заполнение массива
                    }
                }
            }

            return dataset; // Возвращаем тензор с изображениями
        }
    }

}
