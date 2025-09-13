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
    title: "AI-Powered Sales CRM & Call Centre",
    description: "Built scalable CRM with Twilio-powered call centre, AI transcription and analysis using OpenAI. Increased customer conversions by 25% through automated workflows.",
    technologies: ["NextJS", "Twilio", "OpenAI", "PostgreSQL", "AWS", "Vercel"],
    githubUrl: "",
    liveUrl: "",
    featured: true,
  },
  {
    title: "Price Comparison Platform",
    description: "Architected contract management solution with automated workflows, real-time price comparisons, and Experian credit check integration for compliance.",
    technologies: ["React", "Node.js", "Experian API", "PostgreSQL", "Docker", "AWS"],
    githubUrl: "",
    featured: true,
  },
  {
    title: "Cloud-Native Medical Data Platform",
    description: "Led development of highly available, serverless platform for clinical data collection with end-to-end encryption and GDPR compliance.",
    technologies: ["AWS Lambda", "DynamoDB", "API Gateway", "React Native", "GraphQL"],
    githubUrl: "",
    featured: true,
  },
  {
    title: "Data Lakehouse Architecture",
    description: "Designed secure Data Lakehouse for patient data with SCD Type 2 data warehouse, ensuring healthcare compliance and real-time analytics.",
    technologies: ["Apache Airflow", "DBT", "Clickhouse", "Apache Superset", "AWS S3", "Athena"],
    liveUrl: "",
    featured: true,
  },
];

export const experiences: Experience[] = [
  {
    company: "Watt",
    position: "Chief Technology Officer",
    duration: "2021 - Present",
    description: "Leading development teams, ensuring efficient delivery, code quality, and secure cloud infrastructure.",
    achievements: [
      "Built scalable customer data pipelines and centralised SCD Type 2 data warehouse",
      "Architected sales CRM with Twilio-powered call centre, boosting conversions by 25%",
      "Implemented AI-powered call recording, transcription, and analysis with OpenAI",
      "Created compliance workflow with Experian credit checks and automated submissions",
    ],
  },
  {
    company: "Vitaccess",
    position: "Technical Lead",
    duration: "2020 - 2022",
    description: "Led development of a highly available, cloud-native platform for medical data using serverless strategies.",
    achievements: [
      "Managed hiring, cost estimates, documentation, and compliance",
      "Developed clinical survey apps with microservices and end-to-end test suite",
      "Designed secure Data Lakehouse for patient data ensuring compliance",
      "Architected localization API with Wordbee and GraphQL",
    ],
  },
  {
    company: "AFerry",
    position: "Full-Stack Web Developer",
    duration: "2019 - 2020",
    description: "Developed and maintained web applications using modern frameworks and cloud technologies.",
    achievements: [
      "Built responsive web applications with React and Node.js",
      "Implemented RESTful APIs and microservices architecture",
      "Collaborated with cross-functional teams in agile environment",
    ],
  },
  {
    company: "New Orbit",
    position: "Full-Stack Web Developer",
    duration: "2018 - 2019",
    description: "Contributed to client projects and internal product development.",
    achievements: [
      "Developed features for multiple client applications",
      "Worked with modern JavaScript frameworks and cloud services",
      "Participated in code reviews and technical documentation",
    ],
  },
];

export const skills: Skill[] = [
  {
    category: "Current Stack",
    items: [
      { name: "NextJS/Vercel", level: "Expert" },
      { name: "TypeScript/Node", level: "Expert" },
      { name: "Twilio", level: "Expert" },
      { name: "Tailwind/ShadCN", level: "Expert" },
      { name: "Inngest", level: "Advanced" },
    ],
  },
  {
    category: "Cloud & Infrastructure",
    items: [
      { name: "AWS (EC2/ECS/Lambda)", level: "Expert" },
      { name: "Terraform", level: "Expert" },
      { name: "Docker", level: "Expert" },
      { name: "Azure/GCP", level: "Intermediate" },
      { name: "Vercel/Cloudflare", level: "Advanced" },
    ],
  },
  {
    category: "Data & AI",
    items: [
      { name: "AI SDKs (OpenAI/Claude AI)", level: "Advanced" },
      { name: "Apache Superset, DBT, Airflow, Meltano", level: "Intermediate" },
      { name: "Supabase/RDS", level: "Advanced" },
      { name: "PostgreSQL", level: "Expert" },
      { name: "Clickhouse/Athena", level: "Advanced" },
    ],
  },
  {
    category: "Languages & Frameworks",
    items: [
      { name: "React/React Native", level: "Expert" },
      { name: "Python", level: "Advanced" },
      { name: "C#/.NET", level: "Advanced" },
      { name: "GraphQL", level: "Intermediate" },
      { name: "XState", level: "Intermediate" },
    ],
  },
];

export const aboutMe = {
  title: "About Me",
  description: "Fullstack Product Engineer & CTO with 14+ years of experience actively architecting and coding solutions",
  content: `I'm a hands-on Fullstack Product Engineer and CTO with over 14 years of experience in software development and technology leadership. 
  I specialize in building scalable cloud-native platforms, implementing AI/ML solutions, and leading high-performing engineering teams.
  
  My expertise spans AWS cloud architecture, modern web technologies (React, Node.js, TypeScript), and emerging technologies like AI integrations with OpenAI and Claude. I'm passionate about driving technical innovation while maintaining a strong focus on business outcomes and team development.`,
  highlights: [
    { label: "Years of Experience", value: "14+" },
    { label: "Years as CTO/Lead", value: "7+" },
    { label: "Technologies", value: "30+" },
    { label: "Cloud Platforms", value: "AWS, Azure, GCP" },
  ],
};
