export interface Project {
  title: string;
  description: string;
  image?: string;
  technologies: string[];
  githubUrl?: string;
  liveUrl?: string;
  featured: boolean;
}

export interface Experience {
  company: string;
  position: string;
  duration: string;
  description: string;
  achievements: string[];
}

export interface Skill {
  category: string;
  items: {
    name: string;
    level: "Expert" | "Advanced" | "Intermediate";
  }[];
}

export const projects: Project[] = [
  {
    title: "E-Commerce Platform",
    description: "Full-stack e-commerce solution with real-time inventory management, payment processing, and analytics dashboard.",
    technologies: ["Next.js", "TypeScript", "PostgreSQL", "Stripe", "Redis"],
    githubUrl: "https://github.com/gruckion/ecommerce",
    liveUrl: "https://demo.example.com",
    featured: true,
  },
  {
    title: "Task Management System",
    description: "Collaborative project management tool with Kanban boards, real-time updates, and team analytics.",
    technologies: ["React", "Node.js", "MongoDB", "Socket.io", "Docker"],
    githubUrl: "https://github.com/gruckion/taskmanager",
    featured: true,
  },
  {
    title: "DevOps Automation Toolkit",
    description: "Suite of automation tools for CI/CD pipelines, infrastructure provisioning, and monitoring.",
    technologies: ["Python", "Terraform", "AWS", "GitHub Actions", "Prometheus"],
    githubUrl: "https://github.com/gruckion/devops-toolkit",
    featured: true,
  },
  {
    title: "Real-time Analytics Dashboard",
    description: "Data visualization platform for business metrics with customizable widgets and real-time updates.",
    technologies: ["Vue.js", "D3.js", "FastAPI", "ClickHouse", "WebSockets"],
    liveUrl: "https://analytics.example.com",
    featured: true,
  },
];

export const experiences: Experience[] = [
  {
    company: "Tech Solutions Inc.",
    position: "Senior Full-Stack Developer",
    duration: "2022 - Present",
    description: "Leading development of enterprise applications and mentoring junior developers.",
    achievements: [
      "Architected microservices reducing system latency by 40%",
      "Led team of 5 developers on critical client projects",
      "Implemented CI/CD pipelines improving deployment frequency by 3x",
    ],
  },
  {
    company: "Digital Innovations",
    position: "Full-Stack Developer",
    duration: "2020 - 2022",
    description: "Developed and maintained multiple client applications using modern web technologies.",
    achievements: [
      "Built RESTful APIs serving 1M+ requests daily",
      "Reduced application load time by 60% through optimization",
      "Introduced automated testing increasing code coverage to 85%",
    ],
  },
  {
    company: "StartUp Co",
    position: "Junior Developer",
    duration: "2018 - 2020",
    description: "Contributed to various features and learned best practices in software development.",
    achievements: [
      "Developed key features for flagship product",
      "Participated in agile development processes",
      "Improved database query performance by 50%",
    ],
  },
];

export const skills: Skill[] = [
  {
    category: "Frontend",
    items: [
      { name: "React/Next.js", level: "Expert" },
      { name: "TypeScript", level: "Expert" },
      { name: "Vue.js", level: "Advanced" },
      { name: "Tailwind CSS", level: "Expert" },
      { name: "Astro", level: "Advanced" },
    ],
  },
  {
    category: "Backend",
    items: [
      { name: "Node.js", level: "Expert" },
      { name: "Python", level: "Advanced" },
      { name: "PostgreSQL", level: "Expert" },
      { name: "MongoDB", level: "Advanced" },
      { name: "Redis", level: "Advanced" },
    ],
  },
  {
    category: "DevOps & Tools",
    items: [
      { name: "Docker", level: "Advanced" },
      { name: "AWS", level: "Advanced" },
      { name: "CI/CD", level: "Expert" },
      { name: "Git", level: "Expert" },
      { name: "Linux", level: "Advanced" },
    ],
  },
];

export const aboutMe = {
  title: "About Me",
  description: "Passionate full-stack developer with expertise in modern web technologies",
  content: `I'm a full-stack developer with over 6 years of experience building scalable web applications. 
  I specialize in React, Node.js, and cloud technologies, with a strong focus on performance and user experience.
  
  When I'm not coding, I enjoy contributing to open source projects, writing technical articles, and mentoring aspiring developers.`,
  highlights: [
    { label: "Years of Experience", value: "6+" },
    { label: "Projects Completed", value: "50+" },
    { label: "Team Members Led", value: "15+" },
    { label: "Open Source Contributions", value: "100+" },
  ],
};