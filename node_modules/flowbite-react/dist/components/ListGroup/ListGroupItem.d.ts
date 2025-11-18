import { type ComponentProps, type FC } from "react";
import type { FlowbiteBoolean, ThemingProps } from "../../types";
export interface ListGroupItemTheme {
    base: string;
    link: {
        base: string;
        active: FlowbiteBoolean;
        disabled: FlowbiteBoolean;
        href: FlowbiteBoolean;
        icon: string;
    };
}
type GenericLinkButtonProps = ComponentProps<"a"> & ComponentProps<"button">;
export interface ListGroupItemProps extends GenericLinkButtonProps, ThemingProps<ListGroupItemTheme> {
    active?: boolean;
    icon?: FC<ComponentProps<"svg">>;
}
export declare const ListGroupItem: import("react").ForwardRefExoticComponent<Omit<ListGroupItemProps, "ref"> & import("react").RefAttributes<HTMLLIElement>>;
export {};
