import type { ImageMetadata } from 'astro';
import royImage from '@/assets/roy.webp';
import kikiImage from '@/assets/kiki.webp';
import troyImage from '@/assets/troy.webp';
import junoImage from '@/assets/juno.webp';
import javaImage from '@/assets/java.webp';
import myGirlsImage from '@/assets/my_girls.webp';

interface HobbiesProps {
  image: ImageMetadata;
  alt: string;
  orientation: "portrait" | "landscape";
}

export const hobbies: HobbiesProps[] = [
  {
    image: royImage,
    alt: "Roy the dog",
    orientation: "landscape",
  },
  {
    image: kikiImage,
    alt: "Kiki the dog - Achilles",
    orientation: "portrait",
  },
  {
    image: troyImage,
    alt: "Troy the dog",
    orientation: "landscape",
  },
  {
    image: junoImage,
    alt: "Juno the dog",
    orientation: "portrait",
  },
  {
    image: javaImage,
    alt: "Java the dog",
    orientation: "landscape",
  },
  {
    image: myGirlsImage,
    alt: "My girls - Lucy, Java, Juno, and Human",
    orientation: "landscape",
  },
];
