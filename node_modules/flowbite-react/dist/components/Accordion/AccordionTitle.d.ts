import type { ComponentProps, FC } from "react";
import type { FlowbiteBoolean, FlowbiteHeadingLevel, ThemingProps } from "../../types";
export interface AccordionTitleTheme {
    arrow: {
        base: string;
        open: FlowbiteBoolean;
    };
    base: string;
    flush: FlowbiteBoolean;
    heading: string;
    open: FlowbiteBoolean;
}
export interface AccordionTitleProps extends ComponentProps<"button">, ThemingProps<AccordionTitleTheme> {
    arrowIcon?: FC<ComponentProps<"svg">>;
    as?: FlowbiteHeadingLevel;
}
export declare function AccordionTitle(props: AccordionTitleProps): import("react/jsx-runtime").JSX.Element;
export declare namespace AccordionTitle {
    var displayName: string;
}
