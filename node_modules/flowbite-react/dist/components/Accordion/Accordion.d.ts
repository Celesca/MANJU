import type { ComponentProps, FC, ReactElement } from "react";
import type { FlowbiteBoolean, ThemingProps } from "../../types";
import type { AccordionContentTheme } from "./AccordionContent";
import type { AccordionPanelProps } from "./AccordionPanel";
import type { AccordionTitleTheme } from "./AccordionTitle";
export interface AccordionTheme {
    root: AccordionRootTheme;
    content: AccordionContentTheme;
    title: AccordionTitleTheme;
}
export interface AccordionRootTheme {
    base: string;
    flush: FlowbiteBoolean;
}
export interface AccordionProps extends ComponentProps<"div">, ThemingProps<AccordionRootTheme> {
    alwaysOpen?: boolean;
    arrowIcon?: FC<ComponentProps<"svg">>;
    children: ReactElement<AccordionPanelProps> | ReactElement<AccordionPanelProps>[];
    flush?: boolean;
    collapseAll?: boolean;
}
export declare function Accordion(props: AccordionProps): import("react/jsx-runtime").JSX.Element;
export declare namespace Accordion {
    var displayName: string;
}
