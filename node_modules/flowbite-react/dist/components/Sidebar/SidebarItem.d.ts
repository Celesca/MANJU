import type { ComponentProps, ElementType, FC } from "react";
import type { DynamicStringEnumKeysOf, FlowbiteColors, ThemingProps } from "../../types";
export interface SidebarItemTheme {
    active: string;
    base: string;
    collapsed: {
        insideCollapse: string;
        noIcon: string;
    };
    content: {
        base: string;
    };
    icon: {
        base: string;
        active: string;
    };
    label: string;
    listItem: string;
}
export interface SidebarItemProps extends Omit<ComponentProps<"div">, "ref">, ThemingProps<SidebarItemTheme> {
    active?: boolean;
    as?: ElementType;
    href?: string;
    icon?: FC<ComponentProps<"svg">>;
    label?: string;
    labelColor?: DynamicStringEnumKeysOf<SidebarItemLabelColors>;
}
export interface SidebarItemLabelColors extends Pick<FlowbiteColors, "gray"> {
    [key: string]: string;
}
export declare const SidebarItem: import("react").ForwardRefExoticComponent<SidebarItemProps & import("react").RefAttributes<Element>>;
