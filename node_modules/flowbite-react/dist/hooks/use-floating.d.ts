import type { ElementProps, Placement, ReferenceType, UseFloatingReturn, UseRoleProps } from "@floating-ui/react";
import { useFloating } from "@floating-ui/react";
import type { Dispatch, RefObject, SetStateAction } from "react";
export type UseBaseFloatingParams = {
    placement?: "auto" | Placement;
    open: boolean;
    arrowRef?: RefObject<HTMLDivElement>;
    setOpen: Dispatch<SetStateAction<boolean>>;
};
export declare const useBaseFloating: <Type extends ReferenceType>({ open, arrowRef, placement, setOpen, }: UseBaseFloatingParams) => UseFloatingReturn<Type>;
export type UseFloatingInteractionsParams = {
    context: ReturnType<typeof useFloating>["context"];
    trigger?: "hover" | "click";
    role?: UseRoleProps["role"];
    interactions?: ElementProps[];
};
export declare const useFloatingInteractions: ({ context, trigger, role, interactions, }: UseFloatingInteractionsParams) => import("@floating-ui/react").UseInteractionsReturn;
