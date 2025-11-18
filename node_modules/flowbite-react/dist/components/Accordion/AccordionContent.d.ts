import type { ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface AccordionContentTheme {
    base: string;
}
export interface AccordionContentProps extends ComponentProps<"div">, ThemingProps<AccordionContentTheme> {
}
export declare function AccordionContent(props: AccordionContentProps): import("react/jsx-runtime").JSX.Element;
export declare namespace AccordionContent {
    var displayName: string;
}
