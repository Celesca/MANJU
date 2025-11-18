import type { ComponentProps } from "react";
import type { FlowbiteBoolean, ThemingProps } from "../../types";
import type { DrawerHeaderTheme } from "./DrawerHeader";
import type { DrawerItemsTheme } from "./DrawerItems";
export interface DrawerTheme {
    root: DrawerRootTheme;
    header: DrawerHeaderTheme;
    items: DrawerItemsTheme;
}
export interface DrawerRootTheme {
    base: string;
    backdrop: string;
    edge: string;
    position: {
        top: FlowbiteBoolean;
        right: FlowbiteBoolean;
        bottom: FlowbiteBoolean;
        left: FlowbiteBoolean;
    };
}
export interface DrawerProps extends ComponentProps<"div">, ThemingProps<DrawerTheme> {
    backdrop?: boolean;
    edge?: boolean;
    onClose: () => void;
    open?: boolean;
    position?: "top" | "right" | "bottom" | "left";
}
export declare const Drawer: import("react").ForwardRefExoticComponent<Omit<DrawerProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
