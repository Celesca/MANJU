import type { ComponentProps } from "react";
import type { FlowbiteBoolean, ThemingProps } from "../../types";
export interface TabsTheme {
    base: string;
    tablist: {
        base: string;
        variant: TabStyles;
        tabitem: {
            base: string;
            variant: TabStyleItem<TabStyles>;
            icon: string;
        };
    };
    tabitemcontainer: {
        base: string;
        variant: TabStyles;
    };
    tabpanel: string;
}
export interface TabStyles {
    default: string;
    fullWidth: string;
    pills: string;
    underline: string;
}
export interface TabStyleItemProps {
    base: string;
    active: FlowbiteBoolean;
}
export type TabStyleItem<Type> = {
    [K in keyof Type]: TabStyleItemProps;
};
export type TabItemStatus = "active" | "notActive";
export interface TabsProps extends Omit<ComponentProps<"div">, "ref">, ThemingProps<TabsTheme> {
    onActiveTabChange?: (activeTab: number) => void;
    variant?: keyof TabStyles;
}
export interface TabsRef {
    setActiveTab: (activeTab: number) => void;
}
export declare const Tabs: import("react").ForwardRefExoticComponent<TabsProps & import("react").RefAttributes<TabsRef>>;
