import React from 'react';
import { motion } from 'framer-motion';
import { FileText, BookOpen, PenTool } from 'lucide-react';

interface BlogEmptyStateProps {
  delay?: number;
}

const ICON_VARIANTS = {
  left: {
    initial: { scale: 0.8, opacity: 0, x: 0, y: 0, rotate: 0 },
    animate: { scale: 1, opacity: 1, x: 0, y: 0, rotate: -6, transition: { duration: 0.4, delay: 0.1 } },
    hover: { x: -22, y: -5, rotate: -15, scale: 1.1, transition: { duration: 0.2 } }
  },
  center: {
    initial: { scale: 0.8, opacity: 0 },
    animate: { scale: 1, opacity: 1, transition: { duration: 0.4, delay: 0.2 } },
    hover: { y: -10, scale: 1.15, transition: { duration: 0.2 } }
  },
  right: {
    initial: { scale: 0.8, opacity: 0, x: 0, y: 0, rotate: 0 },
    animate: { scale: 1, opacity: 1, x: 0, y: 0, rotate: 6, transition: { duration: 0.4, delay: 0.3 } },
    hover: { x: 22, y: -5, rotate: 15, scale: 1.1, transition: { duration: 0.2 } }
  }
};

export const BlogEmptyState: React.FC<BlogEmptyStateProps> = ({ delay = 0 }) => {
  return (
    <motion.div
      initial={{ y: 40, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, delay }}
      className="group"
    >
      <motion.div
        className="flex flex-col items-center justify-center py-12 text-center"
        initial="initial"
        animate="animate"
        whileHover="hover"
      >
        <div className="mb-6 flex justify-center isolate relative">
          <motion.div
            variants={ICON_VARIANTS.left}
            className="w-12 h-12 rounded-xl flex items-center justify-center relative shadow-lg transition-all duration-300 bg-card border left-2 top-1 z-10"
          >
            <FileText className="h-6 w-6 text-muted-foreground" />
          </motion.div>
          <motion.div
            variants={ICON_VARIANTS.center}
            className="w-12 h-12 rounded-xl flex items-center justify-center relative shadow-lg transition-all duration-300 bg-card border z-20"
          >
            <BookOpen className="h-6 w-6 text-muted-foreground" />
          </motion.div>
          <motion.div
            variants={ICON_VARIANTS.right}
            className="w-12 h-12 rounded-xl flex items-center justify-center relative shadow-lg transition-all duration-300 bg-card border right-2 top-1 z-10"
          >
            <PenTool className="h-6 w-6 text-muted-foreground" />
          </motion.div>
        </div>
        <h3 className="mb-2 text-lg font-semibold text-foreground">No Blogs Right Now</h3>
        <p className="max-w-md text-sm text-muted-foreground">
          Check back later for new articles on development, technology, and career growth.
        </p>
      </motion.div>
    </motion.div>
  );
};