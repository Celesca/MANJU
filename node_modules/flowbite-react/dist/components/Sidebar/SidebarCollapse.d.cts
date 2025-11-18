import type { ComponentProps, FC, ReactElement } from "react";
import type { FlowbiteBoolean, ThemingProps } from "../../types";
import type { SidebarItemProps } from "./SidebarItem";
export interface SidebarCollapseTheme {
    button: string;
    icon: {
        base: string;
        open: FlowbiteBoolean;
    };
    label: {
        base: string;
        title: string;
        icon: {
            base: string;
            open: FlowbiteBoolean;
        };
    };
    list: string;
}
export interface SidebarCollapseProps extends Pick<SidebarItemProps, "active" | "as" | "href" | "icon" | "label" | "labelColor">, ComponentProps<"button">, ThemingProps<SidebarCollapseTheme> {
    onClick?: ComponentProps<"button">["onClick"];
    open?: boolean;
    chevronIcon?: FC<ComponentProps<"svg">>;
    renderChevronIcon?: (theme: SidebarCollapseTheme, open: boolean) => ReactElement;
}
export declare const SidebarCollapse: import("react").ForwardRefExoticComponent<Omit<SidebarCollapseProps, "ref"> & import("react").RefAttributes<HTMLLIElement>>;
