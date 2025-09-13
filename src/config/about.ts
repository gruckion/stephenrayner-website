interface HobbiesProps {
  image: string;
  alt: string;
  orientation: "portrait" | "landscape";
}

export const hobbies: HobbiesProps[] = [
  {
    image: "/roy.jpg",
    alt: "Roy the dog",
    orientation: "landscape",
  },
  {
    image: "/kiki.jpg",
    alt: "Kiki the dog - Achilles",
    orientation: "portrait",
  },
  {
    image: "/troy.jpg",
    alt: "Troy the dog",
    orientation: "landscape",
  },
  {
    image: "/juno.jpg",
    alt: "Juno the dog",
    orientation: "portrait",
  },
  {
    image: "/java.jpg",
    alt: "Java the dog",
    orientation: "landscape",
  },
  {
    image: "/my_girls.jpg",
    alt: "My girls - Lucy, Java, Juno, and Human",
    orientation: "landscape",
  },
];
