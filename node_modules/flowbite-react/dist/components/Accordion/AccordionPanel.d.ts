import type { AccordionProps } from "./Accordion";
export interface AccordionPanelProps extends AccordionProps {
    isOpen?: boolean;
    setOpen?: () => void;
}
export declare function AccordionPanel({ children, ...props }: AccordionPanelProps): import("react/jsx-runtime").JSX.Element;
export declare namespace AccordionPanel {
    var displayName: string;
}
