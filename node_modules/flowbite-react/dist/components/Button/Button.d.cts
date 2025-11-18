import type { ElementType } from "react";
import type { PolymorphicComponentPropWithRef } from "../../helpers/generic-as-prop";
import type { DynamicStringEnumKeysOf, FlowbiteColors, FlowbiteSizes, FlowbiteStateColors, ThemingProps } from "../../types";
export interface ButtonTheme {
    base: string;
    disabled: string;
    fullSized: string;
    grouped: string;
    pill: string;
    size: ButtonSizes;
    color: ButtonColors;
    outlineColor: ButtonOutlineColors;
}
export interface ButtonColors extends Omit<FlowbiteColors, keyof FlowbiteStateColors> {
    [key: string]: string;
    default: string;
    alternative: string;
}
export interface ButtonOutlineColors extends Omit<ButtonColors, "alternative" | "light"> {
    [key: string]: string;
}
export interface ButtonSizes extends Pick<FlowbiteSizes, "xs" | "sm" | "lg" | "xl"> {
    [key: string]: string;
}
export type ButtonProps<T extends ElementType = "button"> = PolymorphicComponentPropWithRef<T, {
    href?: string;
    color?: DynamicStringEnumKeysOf<ButtonColors>;
    fullSized?: boolean;
    outline?: boolean;
    pill?: boolean;
    size?: DynamicStringEnumKeysOf<ButtonSizes>;
}> & ThemingProps<ButtonTheme>;
type ButtonComponentType = (<C extends ElementType = "button">(props: ButtonProps<C>) => JSX.Element) & {
    displayName?: string;
};
export declare const Button: ButtonComponentType;
export {};
