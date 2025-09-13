import type { SidebarNavItem, SiteConfig } from "@/types";

export const siteConfig: SiteConfig = {
  name: "Stephen Rayner",
  description:
    "Personal website of Stephen Rayner - sharing thoughts on technology, development, and career insights.",
  url: "https://stephenrayner.com",
  ogImage: "https://stephenrayner.com/og.jpg",
  links: {
    twitter: "https://x.com/stephen_rayner",
    github: "https://github.com/gruckion",
    linkedin: "https://www.linkedin.com/in/stephen-r-rayner",
  },
};

export const footerLinks: SidebarNavItem[] = [
  {
    title: "Quick Links",
    items: [
      { title: "Home", href: "/" },
      { title: "About", href: "/about" },
      { title: "Blog", href: "/blog" },
      { title: "Newsletter", href: "/newsletter" },
    ],
  },
  {
    title: "Connect",
    items: [
      { title: "GitHub", href: "https://github.com/gruckion", external: true },
      { title: "X (Twitter)", href: "https://x.com/stephen_rayner", external: true },
      { title: "LinkedIn", href: "https://www.linkedin.com/in/stephen-r-rayner", external: true },
    ],
  },
];