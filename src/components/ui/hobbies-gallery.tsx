'use client';

import React from 'react';
import { cn } from '@/lib/utils';
import { useInView } from 'framer-motion';
import { AspectRatio } from '@/components/ui/aspect-ratio';

interface ImageMetadata {
  src: string;
  width: number;
  height: number;
  format: string;
}

interface HobbyImage {
  image: ImageMetadata;
  alt: string;
  orientation: 'portrait' | 'landscape';
}

interface HobbiesGalleryProps {
  hobbies: HobbyImage[];
}

export function HobbiesGallery({ hobbies }: HobbiesGalleryProps) {
  return (
    <div className="relative z-10 mx-auto w-full max-w-4xl px-4 md:max-w-6xl md:px-12 lg:max-w-5xl lg:px-6 xl:max-w-[80rem] xl:px-6 2xl:px-0">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6">
        {hobbies.map((hobby, index) => {
          const ratio = hobby.orientation === 'portrait' ? 3 / 4 : 16 / 9;
          const colSpan = hobby.orientation === 'portrait' ? 'lg:col-span-1 lg:row-span-2' : '';

          return (
            <AnimatedImage
              key={`${hobby.image.src}-${index}`}
              alt={hobby.alt}
              src={hobby.image.src}
              ratio={ratio}
              className={colSpan}
            />
          );
        })}
      </div>
    </div>
  );
}

interface AnimatedImageProps {
  alt: string;
  src: string;
  className?: string;
  ratio: number;
}

function AnimatedImage({ alt, src, ratio, className }: AnimatedImageProps) {
  const ref = React.useRef(null);
  const imgRef = React.useRef<HTMLImageElement>(null);
  const isInView = useInView(ref, { once: true });
  const [isLoading, setIsLoading] = React.useState(true);

  React.useEffect(() => {
    // Handle images that are already loaded before hydration
    if (imgRef.current?.complete) {
      setIsLoading(false);
    }
  }, []);

  return (
    <AspectRatio
      ref={ref}
      ratio={ratio}
      className={cn('relative size-full rounded-xl overflow-hidden', className)}
    >
      <img
        ref={imgRef}
        alt={alt}
        src={src}
        className={cn(
          'size-full rounded-xl object-cover opacity-0 transition-all duration-1000 ease-in-out',
          {
            'opacity-100': isInView && !isLoading,
          },
        )}
        onLoad={() => setIsLoading(false)}
        loading="eager"
      />
    </AspectRatio>
  );
}
