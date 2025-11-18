import type { ComponentProps, FC, ReactNode } from "react";
import type { DynamicStringEnumKeysOf, FlowbiteBoolean, FlowbiteColors, FlowbiteSizes, ThemingProps } from "../../types";
export interface SelectTheme {
    base: string;
    addon: string;
    field: {
        base: string;
        icon: {
            base: string;
            svg: string;
        };
        select: {
            base: string;
            withIcon: FlowbiteBoolean;
            withAddon: FlowbiteBoolean;
            withShadow: FlowbiteBoolean;
            sizes: SelectSizes;
            colors: SelectColors;
        };
    };
}
export interface SelectColors extends Pick<FlowbiteColors, "gray" | "info" | "failure" | "warning" | "success"> {
    [key: string]: string;
}
export interface SelectSizes extends Pick<FlowbiteSizes, "sm" | "md" | "lg"> {
    [key: string]: string;
}
export interface SelectProps extends Omit<ComponentProps<"select">, "color" | "ref">, ThemingProps<SelectTheme> {
    addon?: ReactNode;
    color?: DynamicStringEnumKeysOf<SelectColors>;
    icon?: FC<ComponentProps<"svg">>;
    shadow?: boolean;
    sizing?: DynamicStringEnumKeysOf<SelectSizes>;
}
export declare const Select: import("react").ForwardRefExoticComponent<SelectProps & import("react").RefAttributes<HTMLSelectElement>>;
