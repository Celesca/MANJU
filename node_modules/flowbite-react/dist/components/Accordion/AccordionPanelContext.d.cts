import type { AccordionPanelProps } from "./AccordionPanel";
export type AccordionPanelContextValue = Omit<AccordionPanelProps, "children">;
export declare const AccordionPanelContext: import("react").Context<AccordionPanelContextValue | undefined>;
export declare function useAccordionContext(): AccordionPanelContextValue;
