import { type ComponentProps, type FC } from "react";
import type { DynamicStringEnumKeysOf, FlowbiteBoolean, FlowbiteColors, FlowbiteSizes, ThemingProps } from "../../types";
export interface BadgeTheme {
    root: BadgeRootTheme;
    icon: BadgeIconTheme;
}
export interface BadgeRootTheme {
    base: string;
    color: FlowbiteColors;
    size: BadgeSizes;
}
export interface BadgeIconTheme extends FlowbiteBoolean {
    size: BadgeSizes;
}
export interface BadgeSizes extends Pick<FlowbiteSizes, "xs" | "sm"> {
    [key: string]: string;
}
export interface BadgeProps extends Omit<ComponentProps<"span">, "color">, ThemingProps<BadgeTheme> {
    color?: DynamicStringEnumKeysOf<FlowbiteColors>;
    icon?: FC<ComponentProps<"svg">>;
    size?: DynamicStringEnumKeysOf<BadgeSizes>;
}
export declare const Badge: import("react").ForwardRefExoticComponent<Omit<BadgeProps, "ref"> & import("react").RefAttributes<HTMLSpanElement>>;
