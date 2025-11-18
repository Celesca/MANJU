import type { ComponentProps } from "react";
import type { DynamicStringEnumKeysOf, FlowbiteBoolean, FlowbiteColors, ThemingProps } from "../../types";
export interface TextareaTheme {
    base: string;
    colors: TextareaColors;
    withShadow: FlowbiteBoolean;
}
export interface TextareaColors extends Pick<FlowbiteColors, "gray" | "info" | "failure" | "warning" | "success"> {
    [key: string]: string;
}
export interface TextareaProps extends Omit<ComponentProps<"textarea">, "color" | "ref">, ThemingProps<TextareaTheme> {
    color?: DynamicStringEnumKeysOf<TextareaColors>;
    shadow?: boolean;
}
export declare const Textarea: import("react").ForwardRefExoticComponent<TextareaProps & import("react").RefAttributes<HTMLTextAreaElement>>;
