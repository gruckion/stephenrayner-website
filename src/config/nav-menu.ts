import type { NavMenuConfig } from "@/types";

export const navMenuConfig: NavMenuConfig = {
  pagesNav: [
    {
      title: "Pages",
      items: [
        {
          title: "Home",
          href: "/",
          description: "Welcome to my personal website.",
          image: "/images/examples/landing.jpg",
        },
        {
          title: "About",
          href: "/about",
          description: "Learn more about me and my journey.",
          image: "/images/examples/about.jpg",
        },
        {
          title: "Blog",
          href: "/blog",
          description: "Read my latest articles and thoughts.",
          image: "/images/examples/static-blog.jpg",
        },
        {
          title: "Stack",
          href: "/stack",
          description: "Choose the right tech stack for your next product.",
          image: "/images/examples/stack.jpg",
        },
        // Newsletter disabled for now
        // {
        //   title: "Newsletter",
        //   href: "/newsletter",
        //   description: "Subscribe to stay updated with my latest content.",
        //   image: "/images/examples/newsletter.jpg",
        // },
      ],
    },
  ],
  examplesNav: [],
  links: [],
};
