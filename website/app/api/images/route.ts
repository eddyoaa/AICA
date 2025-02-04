import { AicaEntry, client } from "@/lib/contentful";
import { NextResponse } from "next/server";

export async function GET() {
  try {
    const response = await client.getEntries({
      content_type: "aica",
    });

    const images = response.items.map((item: AicaEntry) => ({
      id: item.sys.id,
      image: {
        url: `https:${item.fields.image.fields.file.url}`,
      },
    }));

    return NextResponse.json({ docs: images });
  } catch (error) {
    console.error("Error fetching AICA images:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
