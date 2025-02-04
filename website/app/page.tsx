"use client";

import { AspectRatio } from "@/components/ui/aspect-ratio";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { placeholderImages } from "@/lib/placeholder-data";
import { Download } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

interface Image {
  id: string;
  title: string;
  image: {
    url: string;
  };
  description: string;
  position: number;
}

export default function Home() {
  const [images, setImages] = useState<Image[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedImage, setSelectedImage] = useState<Image | null>(null);

  useEffect(() => {
    fetch("/api/images")
      .then((res) => res.json())
      .then((data) => {
        const fetchedImages =
          data.docs?.length > 0 ? data.docs : placeholderImages;
        setImages(fetchedImages);
        setLoading(false);
      })
      .catch(() => {
        setImages(placeholderImages);
        setLoading(false);
      });
  }, []);

  const handleDownload = async (image: Image) => {
    try {
      const response = await fetch(image.image.url);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `aica-image-${image.id}.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Error downloading image:", error);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-black to-gray-900 text-white">
      <div className="p-8">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-6xl font-bold mb-24 pb-12 bg-gradient-to-r from-cyan-400 to-yellow-400 bg-clip-text text-transparent">
            AI&apos;m Sitting in a Room
          </h1>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 relative">
            {loading
              ? Array.from({ length: 6 }).map((_, i) => (
                  <Card key={i} className="bg-gray-900 border-gray-800">
                    <Skeleton className="h-[400px] bg-gray-800" />
                  </Card>
                ))
              : images.map((image) => (
                  <Card
                    key={image.id}
                    className="group relative overflow-hidden transition-all duration-300 hover:scale-105 bg-gray-900 border-gray-800 cursor-pointer"
                    onClick={() => setSelectedImage(image)}
                  >
                    <AspectRatio ratio={3 / 4}>
                      <img
                        src={image.image.url}
                        alt={image.title}
                        className="object-cover w-full h-full transition-all duration-500 group-hover:scale-110"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                        <div className="absolute bottom-0 p-6">
                          <h2 className="text-xl text-gray-300 font-bold mb-2">
                            {image.title}
                          </h2>
                          <p className="text-sm text-gray-300">
                            {image.description}
                          </p>
                          <Button
                            variant="secondary"
                            size="sm"
                            className="mt-4"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDownload(image);
                            }}
                          >
                            <Download className="w-4 h-4 mr-2" />
                            Download
                          </Button>
                        </div>
                      </div>
                    </AspectRatio>
                  </Card>
                ))}
          </div>
        </div>
      </div>

      <footer className="border-t border-gray-800 mt-16">
        <div className="max-w-7xl mx-auto px-8 py-6">
          <div className="flex justify-center space-x-6 text-sm text-gray-400">
            <Link
              href="/imprint"
              className="hover:text-white transition-colors"
            >
              Imprint
            </Link>
            <Link
              href="/privacy"
              className="hover:text-white transition-colors"
            >
              Privacy Policy
            </Link>
          </div>
        </div>
      </footer>

      <Dialog
        open={!!selectedImage}
        onOpenChange={() => setSelectedImage(null)}
      >
        <DialogContent className="max-w-[90vw] max-h-[90vh] p-0 bg-black border-gray-800">
          {selectedImage && (
            <>
              <DialogTitle className="sr-only">
                {selectedImage.title}
              </DialogTitle>
              <div className="relative">
                <img
                  src={selectedImage.image.url}
                  alt={selectedImage.title}
                  className="w-full h-full object-contain max-h-[85vh]"
                />
                <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/90 to-transparent">
                  <h2 className="text-2xl font-bold mb-2">
                    {selectedImage.title}
                  </h2>
                  <p className="text-gray-300 mb-4">
                    {selectedImage.description}
                  </p>
                  <Button
                    variant="secondary"
                    onClick={() => handleDownload(selectedImage)}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download Image
                  </Button>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </main>
  );
}
