import type { ComponentProps, FC, ReactNode } from "react";
import type { DynamicStringEnumKeysOf, FlowbiteBoolean, FlowbiteColors, FlowbiteSizes, ThemingProps } from "../../types";
export interface TextInputTheme {
    base: string;
    addon: string;
    field: {
        base: string;
        icon: {
            base: string;
            svg: string;
        };
        rightIcon: {
            base: string;
            svg: string;
        };
        input: {
            base: string;
            sizes: TextInputSizes;
            colors: TextInputColors;
            withIcon: FlowbiteBoolean;
            withRightIcon: FlowbiteBoolean;
            withAddon: FlowbiteBoolean;
            withShadow: FlowbiteBoolean;
        };
    };
}
export interface TextInputColors extends Pick<FlowbiteColors, "gray" | "info" | "failure" | "warning" | "success"> {
    [key: string]: string;
}
export interface TextInputSizes extends Pick<FlowbiteSizes, "sm" | "md" | "lg"> {
    [key: string]: string;
}
export interface TextInputProps extends Omit<ComponentProps<"input">, "ref" | "color">, ThemingProps<TextInputTheme> {
    addon?: ReactNode;
    color?: DynamicStringEnumKeysOf<TextInputColors>;
    icon?: FC<ComponentProps<"svg">>;
    rightIcon?: FC<ComponentProps<"svg">>;
    shadow?: boolean;
    sizing?: DynamicStringEnumKeysOf<TextInputSizes>;
}
export declare const TextInput: import("react").ForwardRefExoticComponent<TextInputProps & import("react").RefAttributes<HTMLInputElement>>;
