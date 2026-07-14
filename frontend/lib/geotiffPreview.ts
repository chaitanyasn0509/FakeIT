import { fromArrayBuffer } from "geotiff";

type RasterBand = ArrayLike<number>;

export async function renderGeoTiffFile(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  return renderGeoTiffBuffer(buffer);
}

export async function renderGeoTiffUrl(url: string): Promise<string> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await response.text());
  const buffer = await response.arrayBuffer();
  return renderGeoTiffBuffer(buffer);
}

async function renderGeoTiffBuffer(buffer: ArrayBuffer): Promise<string> {
  const tiff = await fromArrayBuffer(buffer);
  const image = await tiff.getImage();
  const width = image.getWidth();
  const height = image.getHeight();
  const samples = image.getSamplesPerPixel();
  const selected = samples >= 3 ? [0, 1, 2] : [0];
  const rasters = (await image.readRasters({ samples: selected })) as RasterBand[];
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas rendering is not available.");
  const imageData = context.createImageData(width, height);
  const normalized = rasters.map(normalizeBand);
  for (let index = 0; index < width * height; index += 1) {
    const r = normalized[0][index];
    const g = normalized[1]?.[index] ?? r;
    const b = normalized[2]?.[index] ?? r;
    imageData.data[index * 4] = r;
    imageData.data[index * 4 + 1] = g;
    imageData.data[index * 4 + 2] = b;
    imageData.data[index * 4 + 3] = 255;
  }
  context.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

function normalizeBand(band: RasterBand): Uint8ClampedArray {
  const values = Array.from(band, Number).filter(Number.isFinite).sort((a, b) => a - b);
  const output = new Uint8ClampedArray(band.length);
  if (values.length === 0) return output;
  const low = values[Math.floor(values.length * 0.02)];
  const high = values[Math.floor(values.length * 0.98)];
  const scale = Math.max(high - low, 1e-6);
  for (let index = 0; index < band.length; index += 1) {
    output[index] = Math.max(0, Math.min(255, ((Number(band[index]) - low) / scale) * 255));
  }
  return output;
}
